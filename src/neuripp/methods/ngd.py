import jax.numpy as jnp
import jax.scipy as jsp

from typing import Callable

from flax import nnx
from neuripp._ode._ode import *
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.utility.utility import *


def get_ngd(
    loss: Callable,
    *loss_args,
    step_size: float,
    linear_solver_regularization: float,
    linear_solver_tolerance: float,
    linear_solver_maxiter: float,
    linear_solver_method: str = "cg",
):
    """
    Gives `jax.lax.scan`-compatible function for the Picard/NGD method
    """

    if linear_solver_method != "cg":
        raise NotImplementedError(
            f"Linear solvers except conjugate gradient not supported, but got {linear_solver_method=}"
        )

    vg_fn = nnx.value_and_grad(loss, argnums=0)

    def _ngd_init(_model_ngd, *args, **kwargs):
        _, par, _ = nnx.split(_model_ngd, nnx.Param, ...)
        previous_grad = jax.tree.map(jnp.zeros_like, par)
        return (_model_ngd, previous_grad)

    def _ngd_step(carry, *args):
        _model_ngd = carry[0]
        _previous_grad = carry[1]
        # compute loss and Euclidean grad
        f, grad = vg_fn(_model_ngd, *args)

        # compute natural grad
        matvec_cur = _model_ngd.get_matvec_fn()

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
            _previous_grad,
            tol=linear_solver_tolerance,
            maxiter=linear_solver_maxiter,
        )[0]

        natural_grad_norm_sq = _model_ngd.scalar_product(natural_grad, natural_grad)
        # natural_grad_norm_alt = tree_dot_product(grad, natural_grad)
        grad_norm_sq = tree_dot_product(grad, grad)

        gd, params, rest = nnx.split(_model_ngd, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda x, y: x - y * step_size, params, natural_grad)
        _model_ngd = nnx.merge(gd, params_new, rest)

        return (_model_ngd, natural_grad), (
            f,
            grad_norm_sq,
            natural_grad_norm_sq,
        )

    return _ngd_init, _ngd_step
