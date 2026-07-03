import jax.numpy as jnp

def logpdf_st(x: jnp.ndarray) -> jnp.ndarray:
    """
    Styblinski-Tang potential U(x) = 0.5 * Σ (x_i^4 - 16*x_i^2 + 5*x_i)
    Global minimum at x_i = -2.903534 for all i
    """

    return -0.5 * jnp.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=-1)
