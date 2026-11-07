import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def get_logpdf_ks_inverse_problem(
    num_points=200, num_sensors=20, truth_seed=42, observation_seed=43,
    return_all=False,
):
    """Fixed KS posterior on (..., num_points); optionally return InverseProblem.

    See inverse_common.InverseProblem for the diagnostic dataclass contract.
    Exponax is an optional dependency, loaded only when this getter is called.
    """
    if __package__:
        from .inverse_ks import get_logpdf_ks_inverse_problem as build
    else:
        from inverse_ks import get_logpdf_ks_inverse_problem as build
    return build(num_points, num_sensors, truth_seed, observation_seed, return_all)


def get_logpdf_acoustic_inverse_problem(
    n_grid=256, num_sources=64, num_receivers=64, truth_seed=42,
    observation_seed=43, source_batch_size=1,
    checkpoint_strategy="solver_and_chunk", data_source_batch_size=None,
    return_all=False,
):
    """Fixed acoustic posterior on (..., n_grid**2), with lazy j-Wave import.

    return_all=True returns inverse_common.InverseProblem, including flat truth,
    raw clean/noisy traces and physical prior operators. See inverse_acoustic
    for source chunking, checkpoint modes and resolution-scaling conventions.
    """
    if __package__:
        from .inverse_acoustic import get_logpdf_acoustic_inverse_problem as build
    else:
        from inverse_acoustic import get_logpdf_acoustic_inverse_problem as build
    return build(n_grid, num_sources, num_receivers, truth_seed, observation_seed,
                 source_batch_size, checkpoint_strategy, data_source_batch_size, return_all)


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


def get_logpdf_elliptic_inverse_problem(
    seed: int, dim: int = 6, n_grid: int = 100, noise_std=1e-2, return_all=False
):
    from uncprop.models.elliptic_pde.inverse_problem import (
        generate_pde_inv_prob_rep,
        PDESettings,
    )

    key = jax.random.PRNGKey(seed)
    key_inv_prob, key_surrogate, key_rff = jax.random.split(key, 3)

    # default settings
    n_kl_modes = dim
    # in the original paper these are indicies
    # need to transform them so that the sensor location is the same
    # for every discretization
    obs_x = jnp.array([10, 30, 60, 75]) / 100.0
    obs_idx = jnp.astype(jnp.ceil(obs_x * n_grid), jnp.int32)

    settings = {
        "noise_cov": noise_std**2 * jnp.identity(len(obs_idx)),
        "n_kl_modes": n_kl_modes,
        "obs_locations": obs_idx,
        "settings": PDESettings(n_grid=n_grid),
    }

    # exact posterior
    inv_prob_info = generate_pde_inv_prob_rep(key=key_inv_prob, **settings)
    if return_all:
        return inv_prob_info

    posterior = inv_prob_info[0]

    return posterior.log_density
