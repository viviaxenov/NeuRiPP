from typing import Callable

import jax.numpy as jnp
from flax import nnx

from ..parametric_pushforward.parametric_pushforward import ParametricPushforward

def getKL(logpdf_target: Callable, with_aux: bool = False):

    def _KL(model: ParametricPushforward, latent_batch: jnp.ndarray, rngs: nnx.Rngs, *args):
        x, logpdf = model(latent_batch, rngs, with_log_density=True)
        loss_val = (logpdf - logpdf_target(x)).mean()

        if with_aux:
            # to i need this though?
            return loss_val, (x, logpdf)
        else:
            return loss_val

    return _KL

