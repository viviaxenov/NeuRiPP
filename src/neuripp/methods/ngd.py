import jax.numpy as jnp
import jax.scipy as jsp

from typing import Callable

from flax import nnx
from neuripp._ode._ode import *
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.utility.utility import *


def get_ngd(
    loss: Callable,
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

    vg_fn = nnx.value_and_grad(loss)

    def _ngd_init(_model_ngd, *args, **kwargs):
        return (_model_ngd,)

    def _ngd_step(carry, *args):
        _model_ngd = carry[0]
        # compute loss and Euclidean grad
        f, grad = vg_fn(_model_ngd)
        matvec_cur = _model_ngd.get_matvec_fn()

        # compute natural grad
        def oper_cg(tang):
            Gtang = matvec_cur(tang)
            return jax.tree.map(
                lambda _g, _x: _g + linear_solver_regularization * _x, Gtang, tang
            )

        natural_grad = jsp.sparse.linalg.cg(
            oper_cg, grad, tol=linear_solver_tolerance, maxiter=linear_solver_maxiter
        )[0]

        natural_grad_norm_sq = _model_ngd.scalar_product(natural_grad, natural_grad)
        # natural_grad_norm_alt = tree_dot_product(grad, natural_grad)
        grad_norm_sq = tree_dot_product(grad, grad)

        gd, params, rest = nnx.split(_model_ngd, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda x, y: x - y * step_size, params, natural_grad)
        _model_ngd = nnx.merge(gd, params_new, rest)

        return (_model_ngd,), (
            f,
            grad_norm_sq,
            natural_grad_norm_sq,
        )

    return _ngd_init, _ngd_step
