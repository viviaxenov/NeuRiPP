from flax import nnx
import optax
from ..parametric_pushforward.parametric_pushforward import ParametricPushforward
from ..utility.utility import tree_dot_product
from typing import Callable


optax_optimizers = {
    "sgd": optax.sgd,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "rmsprop": optax.rmsprop,
    "adagrad": optax.adagrad,
    "yogi": optax.yogi,
    "lion": optax.lion,
    "lbfgs": optax.lbfgs,
}


def get_optax(
    loss: Callable,
    method: str,
    lr: float,
    **optimizer_kwargs,
):
    vg_fn = nnx.value_and_grad(loss, argnums=0)

    def _init(model, ):
        optax_method = optax_optimizers[method](learning_rate=lr, **optimizer_kwargs)
        optimizer = nnx.Optimizer(model, optax_method, wrt=nnx.Param)

        return (model, optimizer)

    def _step(carry, *args):
        model, optimizer = carry

        f, grad = vg_fn(model, *args)

        grad_norm_sq = tree_dot_product(grad, grad)
        optimizer.update(model, grad)

        return (model, optimizer), (
            f,
            grad_norm_sq,
        )

    return _init, _step



