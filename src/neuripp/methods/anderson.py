from typing import Callable, Literal

import jax
import jax.numpy as jnp

from flax import nnx

from ..parametric_pushforward.parametric_pushforward import ParametricPushforward
from ..utility.utility import tree_dot_product
from .ngd import _compute_natural_grad, _clip_gradient

pairwise_dot_col = jax.vmap(tree_dot_product, in_axes=[0, None])
pairwise_dot_matrix = jax.vmap(pairwise_dot_col, in_axes=[None, 0], out_axes=1)


def _update_history(history, residual, delta_x, history_length: int):
    # move vectors to the right to free space
    history = jax.tree.map(lambda _h: jnp.roll(_h, 1, 0), history)
    # write new residual
    history = jax.tree.map(lambda _h, _r: _h.at[0, ...].set(_r), history, residual)
    # write new delta_x
    history = jax.tree.map(
        lambda _h, _x: _h.at[history_length, ...].set(_x), history, delta_x
    )

    return history


def get_anderson(
    loss: Callable,
    history_length: int = 8,
    regularization_method: Literal["l2", "adaptive"] = "l2",
    ensure_descent: bool = True,
    linear_solver_method: str = "cg",
    natural_grad_clipping_threshold: float = None,
    stepsize_schedule_fn: Callable = None,
):
    if linear_solver_method != "cg":
        raise NotImplementedError(
            f"Linear solvers except conjugate gradient not supported yet, but got {linear_solver_method=}"
        )

    vg_fn = nnx.value_and_grad(loss, argnums=0)

    def _init(
        model: ParametricPushforward,
        args,
        kwargs,
        batch,
        rngs,
    ):
        step_size = args[0]

        if stepsize_schedule_fn is not None:
            step_size = stepsize_schedule_fn(step_size, 0, **kwargs)

        gd, par, rest = nnx.split(model, nnx.Param, ...)
        zero_vector = jax.tree.map(jnp.zeros_like, par)
        f, grad = vg_fn(model, batch, rngs)
        natural_grad = _compute_natural_grad(
            model,
            rngs,
            grad,
            zero_vector,
            **kwargs,
        )
        natural_grad_norm_sq = model.scalar_product(natural_grad, natural_grad, rngs)
        norm = jnp.maximum(natural_grad_norm_sq, 0.0) ** 0.5
        # Gradient clipping
        if natural_grad_clipping_threshold is not None:
            natural_grad = _clip_gradient(
                natural_grad, norm, natural_grad_clipping_threshold
            )

        residual = jax.tree.map(lambda _x: -step_size * _x, natural_grad)
        # initialize history with zero vectors
        # history[:m] is residuals
        # history[m:] is Delta x
        # (slice in each leaf of pytree)

        # Do one Picard step
        par = jax.tree.map(lambda _p, _dp: _p + _dp, par, residual)
        model = nnx.merge(gd, par, rest)

        history = jax.tree.map(
            lambda _l: jnp.zeros((2 * history_length, *_l.shape)), residual
        )
        history = _update_history(history, residual, residual, history_length)
        args = (step_size, *args[1:])
        kwargs = {'i_warmstart': 0} | kwargs
        return (model, natural_grad, history, 0, args, kwargs)

    def _step(state, batch, rngs):
        model, previous_grad, history, i, args, kwargs = state
        i_warmstart = kwargs['i_warmstart']

        step_size, relaxation, regularization_factor = args

        if stepsize_schedule_fn is not None:
            step_size = stepsize_schedule_fn(step_size, i, **kwargs)

        f, grad = vg_fn(model, batch, rngs)
        natural_grad = _compute_natural_grad(
            model,
            rngs,
            grad,
            previous_grad,
            **kwargs,
        )

        grad_norm_sq = tree_dot_product(grad, grad)
        # Gradient clipping
        natural_grad_norm_sq = model.scalar_product(natural_grad, natural_grad, rngs)
        norm = jnp.maximum(natural_grad_norm_sq, 0.0) ** 0.5
        if natural_grad_clipping_threshold is not None:
            natural_grad = _clip_gradient(
                natural_grad, norm, natural_grad_clipping_threshold
            )
        natural_grad_norm_sq = model.scalar_product(natural_grad, natural_grad, rngs)
        norm = jnp.maximum(natural_grad_norm_sq, 0.0) ** 0.5

        residual = jax.tree.map(lambda _x: -step_size * _x, natural_grad)
        r_cur_norm_sq = natural_grad_norm_sq * step_size**2
        # compute previous delta_r
        history = jax.tree.map(
            lambda _h, _r: _h.at[0].set(_r - _h[0]), history, residual
        )
        # precompute Gx for historical vectors
        # TODO: this requires a backward pass, maybe better use scalar product with forward passes only?
        # TODO: use nnx.vmap?
        Ghistory = nnx.vmap(model.get_matvec_fn(rngs))(history)
        # compute dot products in parallel
        # <r_i, Gr_j>
        pairwise_mat = pairwise_dot_matrix(history, Ghistory)
        rhs = pairwise_dot_col(
            jax.tree.map(lambda _h: _h[:history_length, ...], Ghistory), residual
        )
        # assemble small matrices for the subprobplem
        RR = pairwise_mat[:history_length, :history_length]
        if regularization_method == "adaptive":
            XX = pairwise_mat[history_length:, history_length:]
            delta_x_norm_sq = XX[0, 0]

            rf = regularization_factor * r_cur_norm_sq / (delta_x_norm_sq + 1e-8)
            # without epsilon*Id, we get nan = 0/0 error at steps < history_length
            # when delta_r and delta_x are not yet initialized
            M_reg = XX + jnp.eye(history_length) * 1e-20
        else:
            rf = regularization_factor
            M_reg = jnp.eye(history_length)

        S = RR + rf * M_reg
        S = 0.5 * (S + S.T)

        gamma = jnp.linalg.solve(S, rhs)

        mixing_weights = jnp.concatenate((relaxation * gamma, gamma))
        r_mixed = jax.tree.map(
            lambda _l: jnp.einsum("i...,i->...", _l, mixing_weights), history
        )

        if ensure_descent:
            descending = (
                model.scalar_product(residual, r_mixed, rngs)
                <= relaxation * r_cur_norm_sq
            )
            r_mixed = jax.tree.map(
                lambda _dr: jnp.where(descending, _dr, _dr * 0.0), r_mixed
            )

        # if warmstart, ignore anderson correction
        r_mixed = jax.tree.map(
            lambda _dr: jnp.where(i > i_warmstart, _dr, _dr * 0.0), r_mixed
        )

        delta_x = jax.tree.map(lambda _r, _dr: relaxation * _r - _dr, residual, r_mixed)

        history = _update_history(
            history, residual, delta_x, history_length=history_length
        )
        # don't write history if in warmup stage
        history = jax.tree.map(lambda _h : jnp.where(i > i_warmstart, _h, _h.at[1:history_length, ...].set(0.)), history)
        history = jax.tree.map(lambda _h : jnp.where(i > i_warmstart, _h, _h.at[history_length + 1:, ...].set(0.)), history)

        gd, params, rest = nnx.split(model, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda _x, _dx: _x + _dx, params, delta_x)
        model = nnx.merge(gd, params_new, rest)

        args = (step_size, *args[1:])

        return (model, natural_grad, history, i + 1, args, kwargs), (
            f,
            grad_norm_sq,
            natural_grad_norm_sq,
        )

    return _init, _step
