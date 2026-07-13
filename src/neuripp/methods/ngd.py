import jax.numpy as jnp
import jax.scipy as jsp

from typing import Callable

from flax import nnx
from neuripp._ode._ode import *
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.utility.utility import *


def _clip_gradient(natural_grad, snorm: float, max_snorm: float):
    factor = jnp.minimum(1.0, max_snorm / snorm)
    return jax.tree.map(lambda _x: _x * factor, natural_grad)


def _compute_natural_grad(
    model,
    rngs,
    grad,
    init_vector,
    linear_solver_regularization: float = 1e-3,
    linear_solver_tolerance: float = 1e-6,
    linear_solver_maxiter: float = 50,
):
    matvec_cur = model.get_matvec_fn(rngs)

    def oper_cg(tang):
        Gtang = matvec_cur(tang)
        return jax.tree.map(
            lambda _g, _x: _g + linear_solver_regularization * _x,
            Gtang,
            tang,
        )

    natural_grad = jsp.sparse.linalg.cg(
        oper_cg,
        grad,
        init_vector,
        tol=linear_solver_tolerance,
        maxiter=linear_solver_maxiter,
    )[0]

    return natural_grad


def get_ngd(
    loss: Callable,
    linear_solver_method: str = "cg",
    natural_grad_clipping_threshold: float = None,
):
    """
    Gives `jax.lax.scan`-compatible function for the Picard/NGD method
    """
    if linear_solver_method != "cg":
        raise ValueError(
            f"Only linear_solver_method='cg' is implemented so far, but got {linear_solver_method}"
        )
    vg_fn = nnx.value_and_grad(loss, argnums=0)

    def _ngd_init(model, args, kwargs, *rest_args):
        _, par, _ = nnx.split(model, nnx.Param, ...)
        previous_grad = jax.tree.map(jnp.zeros_like, par)
        state = (model, previous_grad, args, kwargs)
        return state

    def _ngd_step(
        state,
        batch,
        rngs,
    ):
        model, prev_grad, args, kwargs = state
        step_size = args[0]
        # compute loss and Euclidean grad
        f, grad = vg_fn(model, batch, rngs)

        # compute natural grad
        # natural_grad_norm_alt = tree_dot_product(grad, natural_grad)
        natural_grad = _compute_natural_grad(
            model,
            rngs,
            grad,
            prev_grad,
            **kwargs,
        )
        grad_norm_sq = tree_dot_product(grad, grad)

        natural_grad_norm_sq = model.scalar_product(natural_grad, natural_grad, rngs)
        norm = jnp.maximum(natural_grad_norm_sq, 0.0) ** 0.5
        # Gradient clipping
        if natural_grad_clipping_threshold is not None:
            natural_grad = _clip_gradient(
                natural_grad, norm * step_size, natural_grad_clipping_threshold
            )

        gd, params, rest = nnx.split(model, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda x, y: x - y * step_size, params, natural_grad)
        model = nnx.merge(gd, params_new, rest)

        return (model, natural_grad, args, kwargs), (
            f,
            grad_norm_sq,
            natural_grad_norm_sq,
        )

    return _ngd_init, _ngd_step
