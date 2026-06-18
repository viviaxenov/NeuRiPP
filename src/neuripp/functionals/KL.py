from typing import Callable

import jax.numpy as jnp

from ..parametric_pushforward.parametric_pushforward import ParametricPushforward

def getKL(logpdf_target: Callable, batch_size: int, with_aux: bool = False):

    def _KL(model: ParametricPushforward, *args):
        x, logpdf = model.sample(batch_size, with_log_density=True)
        loss_val = (logpdf - logpdf_target(x)).mean()

        if with_aux:
            return loss_val, (x, logpdf)
        else:
            return loss_val

    return _KL


def get_logpdf_double_banana():
    pass



def logpdf_st(x: jnp.ndarray) -> jnp.ndarray:
    """
    Styblinski-Tang potential U(x) = 0.5 * Σ (x_i^4 - 16*x_i^2 + 5*x_i)
    Global minimum at x_i = -2.903534 for all i
    """
    return -0.5 * jnp.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=-1)
