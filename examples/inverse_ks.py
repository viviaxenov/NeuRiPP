"""Conservative Exponax KS initial-condition inverse benchmark."""

import jax
import jax.numpy as jnp

if __package__:
    from .inverse_common import InverseProblem, fft_prior, flat_batch, positive_int
else:
    from inverse_common import InverseProblem, fft_prior, flat_batch, positive_int


def get_logpdf_ks_inverse_problem(
    num_points=200, num_sensors=20, truth_seed=42, observation_seed=43,
    return_all=False,
):
    """Build a fixed KS target; return an InverseProblem when return_all=True.

    Parameters are flat physical initial conditions, not whitened latents.
    Optional Exponax is imported only when constructing this target.
    """
    num_points = positive_int("num_points", num_points)
    num_sensors = positive_int("num_sensors", num_sensors)
    if num_points < 4 or num_sensors > num_points:
        raise ValueError("require num_points >= 4 and num_sensors <= num_points")
    import exponax as ex

    stepper = ex.stepper.KuramotoSivashinskyConservative(
        num_spatial_dims=1, domain_extent=100.0, num_points=num_points,
        dt=0.1, order=2,
    )
    sensors = jnp.arange(num_sensors) * num_points // num_sensors

    def forward_single(x):
        # Save only the four observation states, rematerialize each interval
        # for reverse mode rather than retaining the complete trajectory.
        def interval(state, _):
            state = jax.lax.fori_loop(0, 50, lambda i, u: stepper(u), state)
            return state, state[0, sensors]
        return jax.lax.scan(jax.checkpoint(interval), x[None, :], None, length=4)[1]

    forward = flat_batch(forward_single, num_points)
    apply, inverse, logprior = fft_prior((num_points,), (100 / num_points,), 2., 1.)
    truth = apply(jax.random.normal(jax.random.PRNGKey(truth_seed), (num_points,)))
    clean = forward(truth)
    noise_std = jnp.asarray(0.1)
    noisy = clean + noise_std * jax.random.normal(
        jax.random.PRNGKey(observation_seed), clean.shape)
    likelihood_single = lambda x: -0.5 * jnp.sum(((forward_single(x) - noisy) / noise_std)**2)
    likelihood = flat_batch(likelihood_single, num_points)
    logpdf = flat_batch(lambda x: logprior(x) + likelihood_single(x), num_points)
    result = InverseProblem(
        logpdf, logprior, likelihood, forward, apply, inverse, truth, clean,
        noisy, noise_std,
        dict(num_points=num_points, num_sensors=num_sensors, sensor_indices=sensors,
             domain_extent=100., dt=0.1, order=2, observation_times=(5., 10., 15., 20.),
             observation_steps=(50, 100, 150, 200), ell=2., alpha=2., sigma_x=1.,
             truth_seed=truth_seed, observation_seed=observation_seed),
    )
    return result if return_all else result.logpdf
