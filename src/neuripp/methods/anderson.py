from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from flax import nnx

from functools import partial

from ..parametric_pushforward.parametric_pushforward import ParametricPushforward
from ..utility.utility import tree_dot_product
from .ngd import _compute_natural_grad

pairwise_dot_col = jax.vmap(tree_dot_product, in_axes=[0, None])
pairwise_dot_matrix = jax.vmap(pairwise_dot_col, in_axes=[None, 0], out_axes=1)


def _update_history(history, residual, delta_x, history_length: int):
    # move vectors to the right to free space
    history = jax.tree.map(lambda _h: jnp.roll(_h, 1, 0), history)
    # write new residual
    history = jax.tree.map(lambda _h, _r: _h.at[0].set(_r), history, residual)
    # write new delta_x
    history = jax.tree.map(lambda _h, _x: _h.at[history_length].set(_x), history, delta_x)

    return history


def get_anderson(
    loss: Callable,
    step_size: float,
    history_length: int,
    relaxation: float,
    regularization_factor: float,
    regularization_method: Literal["l2", "adaptive"],
    ensure_descent: bool,
    linear_solver_regularization: float,
    linear_solver_tolerance: float,
    linear_solver_maxiter: float,
    linear_solver_method: str = "cg",
):
    if linear_solver_method != "cg":
        raise NotImplementedError(
            f"Linear solvers except conjugate gradient not supported, but got {linear_solver_method=}"
        )

    vg_fn = nnx.value_and_grad(loss, argnums=0)

    # TODO: init w/o calling loss
    def _init(
        model: ParametricPushforward,
        *loss_args,
    ):
        _, par, _ = nnx.split(model, nnx.Param, ...)
        zero_vector = jax.tree.map(jnp.zeros_like, par)
        f, grad = vg_fn(model, *loss_args)
        natural_grad = _compute_natural_grad(
            model,
            grad,
            zero_vector,
            linear_solver_regularization,
            linear_solver_tolerance,
            linear_solver_maxiter,
        )
        # initialize history with zero vectors
        # history[:m] is residuals
        # history[m:] is Delta x
        # (slice in each leaf of pytree)
        residual = jax.tree.map(lambda _x: -step_size * _x, natural_grad)
        history = jax.tree.map(
            lambda _l: jnp.zeros((2 * history_length, *_l.shape)), residual
        )
        # history = _update_history(history, residual, residual, history_length)
        return (model, natural_grad, history)

    def _step(carry, *loss_args):
        model: ParametricPushforward
        model, previous_grad, history = carry
        f, grad = vg_fn(model, *loss_args)
        natural_grad = _compute_natural_grad(
            model,
            grad,
            previous_grad,
            linear_solver_regularization,
            linear_solver_tolerance,
            linear_solver_maxiter,
        )
        residual = jax.tree.map(lambda _x: -step_size * _x, natural_grad)
        r_cur_norm_sq = model.scalar_product(residual, residual)
        # compute previous delta_r
        history = jax.tree.map(
            lambda _h, _r: _h.at[0].set(_r - _h[0]), history, residual
        )
        # precompute Gx for historical vectors
        # TODO: this requires a backward pass, maybe better use scalar product with forward passes only?
        Ghistory = jax.vmap(model.get_matvec_fn())(history)
        # compute dot products in parallel
        # <r_i, Gr_j>
        pairwise_mat = pairwise_dot_matrix(history, Ghistory)
        rhs = pairwise_dot_col(
            jax.tree.map(lambda _l: _l[:history_length, ...], Ghistory), residual
        )
        # assemble small matrices for the subprobplem
        RR = pairwise_mat[:history_length, :history_length]
        if regularization_method == "adaptive":
            XX = pairwise_mat[history_length:, history_length:]
            delta_x_norm_sq = XX[0, 0]

            rf = regularization_factor * r_cur_norm_sq / (delta_x_norm_sq + 1e-8)
            M_reg = XX
        else:
            rf = regularization_factor
            M_reg = jnp.eye(history_length)

        gamma = jnp.linalg.solve(RR + rf * M_reg, rhs)

        mixing_weights = jnp.concatenate((relaxation * gamma, gamma))
        r_mixed = jax.tree.map(
            lambda _l: jnp.einsum("i...,i->...", _l, mixing_weights), history
        )

        if ensure_descent:
            descending = (
                model.scalar_product(residual, r_mixed) <= relaxation * r_cur_norm_sq
            )
            r_mixed = jax.tree.map(
                lambda _dr: jnp.where(descending, _dr, _dr * 0.0), r_mixed
            )
        delta_x = jax.tree.map(lambda _r, _dr: relaxation * _r - _dr, residual, r_mixed)

        history = _update_history(
            history, residual, delta_x, history_length=history_length
        )

        gd, params, rest = nnx.split(model, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda _x, _dx: _x + _dx, params, delta_x)
        model = nnx.merge(gd, params_new, rest)

        grad_norm_sq = tree_dot_product(grad, grad)

        return (model, natural_grad, history), (
            f,
            grad_norm_sq,
            r_cur_norm_sq / step_size**2,
        )

    return _init, _step
