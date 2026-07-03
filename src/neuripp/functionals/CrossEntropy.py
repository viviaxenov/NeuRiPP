import jax.numpy as jnp
from flax import nnx
from ..parametric_pushforward.parametric_pushforward import ParametricPushforward


def cross_entropy(model: ParametricPushforward, data_batch: jnp.ndarray, rngs: nnx.Rngs):
    _, log_pdf_model = model.pullback(data_batch, rngs, with_log_density=True)

    return -log_pdf_model.mean()
