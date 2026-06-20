import jax.numpy as jnp
from ..parametric_pushforward.parametric_pushforward import ParametricPushforward


def cross_entropy(model: ParametricPushforward, data_batch: jnp.ndarray):
    _, log_pdf_model = model.pullback(data_batch, with_log_density=True)

    return -log_pdf_model.mean()
