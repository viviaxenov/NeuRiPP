import jax.numpy as jnp
import jax.scipy as jsp

from typing import Callable

from flax import nnx
from neuripp._ode._ode import *
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.utility.utility import *


def get_sgd(
    loss: Callable,
    step_size: float,
):
    """
    Gives `jax.lax.scan`-compatible function for the Picard/NGD method
    """
    vg_fn = nnx.value_and_grad(loss)

    def _sgd_init(*args, **kwargs):
        return args

    def _sgd_step(carry, *args):
        _model_sgd = carry[0]
        # compute loss and Euclidean grad
        f, grad = vg_fn(_model_ngd)

        grad_norm_sq = tree_dot_product(grad, grad)

        gd, params, rest = nnx.split(_model_ngd, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda x, y: x - y * step_size, params, grad)
        _model_ngd = nnx.merge(gd, params_new, rest)

        return _model_ngd, (
            f,
            grad_norm_sq,
        )

    return _sgd_init, _sgd_step



