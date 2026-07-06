import jax.numpy as jnp
from jax.scipy.special import logsumexp

def logpdf_st(x: jnp.ndarray) -> jnp.ndarray:
    """
    Styblinski-Tang potential U(x) = 0.5 * Σ (x_i^4 - 16*x_i^2 + 5*x_i)
    Global minimum at x_i = -2.903534 for all i
    """

    return -0.5 * jnp.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=-1)


def logpdf_double_banana(x: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
    x_shifted = x - shift
    x1 = x_shifted[..., 0]
    log_density = -2.0 * (jnp.linalg.norm(x_shifted, axis=-1) - 3.0) ** 2
    log_density += logsumexp(
        jnp.stack((-2.0 * (x1 - 3.0) ** 2, -2.0 * (x1 + 3.0) ** 2), axis=-1),
        axis=-1,
    )
    return log_density
