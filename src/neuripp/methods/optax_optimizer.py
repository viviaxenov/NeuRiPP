from flax import nnx
import optax
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
    method: str = "adamw",
):
    vg_fn = nnx.value_and_grad(loss, argnums=0)

    def _init(
        model,
        optimizer_args,
        optimizer_kwargs,
        *args
    ):
        # this allows to vmap the method for multiple learning rates
        optax_method = optax.inject_hyperparams(optax_optimizers[method])(*optimizer_args, **optimizer_kwargs)
        optimizer = nnx.Optimizer(model, optax_method, wrt=nnx.Param)
        # optimizer.opt_state.hyperparams["learning_rate"] = learning_rate
        return (model, optimizer, )

    def _step(
        state,
        batch,
        rngs,
        *args
    ):
        model, optimizer = state

        f, grad = vg_fn(model, batch, rngs)

        grad_norm_sq = tree_dot_product(grad, grad)
        optimizer.update(model, grad)

        return (model, optimizer ), (
            f,
            grad_norm_sq,
        )

    return _init, _step
