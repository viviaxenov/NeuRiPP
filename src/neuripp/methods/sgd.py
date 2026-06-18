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
    Gives `jax.lax.scan`-compatible function for the SGD method
    """
    vg_fn = nnx.value_and_grad(loss, argnums=0)

    def _sgd_init(model, *args, **kwargs):
        return (model,)

    def _sgd_step(carry, *args):
        _model = carry[0]
        # compute loss and Euclidean grad
        f, grad = vg_fn(_model, *args)

        grad_norm_sq = tree_dot_product(grad, grad)

        gd, params, rest = nnx.split(_model, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda x, y: x - y * step_size, params, grad)
        _model = nnx.merge(gd, params_new, rest)

        return (_model,), (
            f,
            grad_norm_sq,
        )

    return _sgd_init, _sgd_step
