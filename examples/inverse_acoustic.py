"""j-Wave raw-trace acoustic inverse benchmark with bounded source memory."""

import numpy as np
import jax
import jax.numpy as jnp

if __package__:
    from .inverse_common import InverseProblem, fft_prior, flat_batch, positive_int
else:
    from inverse_common import InverseProblem, fft_prior, flat_batch, positive_int


def validate_positions(positions, n_grid, pml_size, name):
    """Reject discretized acquisition points outside the interior or repeated."""
    points = np.asarray(positions).T
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(f"invalid {name} positions")
    if not np.all(points == np.floor(points)):
        raise ValueError(f"{name} positions must be integer grid indices")
    if np.any(points < pml_size) or np.any(points >= n_grid - pml_size):
        raise ValueError(f"{name} positions lie outside the non-PML grid")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"duplicate {name} positions; reduce count or increase n_grid")


def get_logpdf_acoustic_inverse_problem(
    n_grid=256, num_sources=64, num_receivers=64, truth_seed=42,
    observation_seed=43, source_batch_size=1,
    checkpoint_strategy="solver_and_chunk", data_source_batch_size=None,
    return_all=False,
):
    """Build the physical-field posterior, or its documented InverseProblem.

    The 25.6 mm domain is fixed as resolution changes. Circle radii and PML
    width scale in grid units (PML rounded to nearest integer, minimum one).
    Source chunks use vmap internally and sequential scans externally; a final
    short chunk is solved exactly once, without padding/duplicating sources.
    Checkpoint modes select solver timestep and/or entire chunk rematerialization.
    data_source_batch_size affects synthetic-data generation only.
    """
    n_grid = positive_int("n_grid", n_grid)
    num_sources = positive_int("num_sources", num_sources)
    num_receivers = positive_int("num_receivers", num_receivers)
    source_batch_size = positive_int("source_batch_size", source_batch_size)
    data_source_batch_size = source_batch_size if data_source_batch_size is None else positive_int(
        "data_source_batch_size", data_source_batch_size)
    if n_grid < 8:
        raise ValueError("n_grid must be at least 8")
    if checkpoint_strategy not in ("none", "solver", "chunk", "solver_and_chunk"):
        raise ValueError("unknown checkpoint_strategy")

    from jwave import FourierSeries
    from jwave.acoustics.time_varying import simulate_wave_propagation, TimeWavePropagationSettings
    from jwave.geometry import Domain, Medium, Sources, Sensors, TimeAxis, points_on_circle, circ_mask
    from jwave.signal_processing import apply_ramp, gaussian_window

    spacing = 25.6e-3 / n_grid
    scale = n_grid / 256
    pml_size = max(1, int(round(20 * scale)))
    center = (n_grid / 2, n_grid / 2)
    source_positions = points_on_circle(num_sources, 100 * scale, center)
    receiver_positions = points_on_circle(num_receivers, 100 * scale, center)
    validate_positions(source_positions, n_grid, pml_size, "source")
    validate_positions(receiver_positions, n_grid, pml_size, "receiver")
    source_positions = tuple(jnp.asarray(p, dtype=jnp.int32) for p in source_positions)
    domain = Domain((n_grid, n_grid), (spacing, spacing))
    sensors = Sensors(positions=receiver_positions)
    mask = jnp.asarray(circ_mask(domain.N, 80 * scale, center))
    time_axis = TimeAxis.from_medium(
        Medium(domain, sound_speed=1630.0), cfl=0.25, t_end=25e-6)
    t = time_axis.to_array()
    pulse = (1.3e-5 / time_axis.dt) * jnp.sin(2 * jnp.pi * 1e6 * t)
    pulse = gaussian_window(apply_ramp(pulse, time_axis.dt, 1e6), t, 3e-6, 1.2e-6)
    solver_checkpoint = checkpoint_strategy in ("solver", "solver_and_chunk")
    chunk_checkpoint = checkpoint_strategy in ("chunk", "solver_and_chunk")
    solver_settings = TimeWavePropagationSettings(checkpoint=solver_checkpoint)

    def single_source(x, index):
        speed = 1480 * (1 + 0.10 * mask * jnp.tanh(x.reshape(domain.N) / 0.10))
        medium = Medium(domain, sound_speed=FourierSeries(speed[..., None], domain), pml_size=pml_size)
        positions = tuple(jax.lax.dynamic_slice(p, (index,), (1,)) for p in source_positions)
        sources = Sources(positions, pulse[None, :], dt=time_axis.dt, domain=domain)
        return simulate_wave_propagation(
            medium, time_axis, sources=sources, sensors=sensors,
            settings=solver_settings)[..., 0]

    chunk = jax.vmap(single_source, in_axes=(None, 0))
    if chunk_checkpoint:
        chunk = jax.checkpoint(chunk)

    def chunks(size):
        size = min(size, num_sources)
        end = num_sources // size * size
        return jnp.arange(end).reshape((-1, size)), jnp.arange(end, num_sources)

    def forward_single(x, size=source_batch_size):
        indices, remainder = chunks(size)
        traces = jax.lax.map(lambda ids: chunk(x, ids), indices)
        traces = traces.reshape((-1,) + traces.shape[2:])
        if remainder.size:
            traces = jnp.concatenate((traces, chunk(x, remainder)), axis=0)
        return traces

    dim = n_grid**2
    forward = flat_batch(forward_single, dim)
    apply, inverse, logprior = fft_prior(domain.N, domain.dx, 1e-3, 0.04)
    truth = apply(jax.random.normal(jax.random.PRNGKey(truth_seed), (dim,)))
    clean = jax.jit(lambda x: forward_single(x, data_source_batch_size))(truth)
    noise_std = 0.02 * jnp.median(jnp.max(jnp.abs(clean), axis=1))
    if not np.isfinite(float(noise_std)) or float(noise_std) <= 0:
        raise ValueError("acquisition produced a nonpositive/nonfinite noise scale")
    noisy = clean + noise_std * jax.random.normal(jax.random.PRNGKey(observation_seed), clean.shape)
    indices, remainder = chunks(source_batch_size)

    def chunk_loss(x, ids):
        return -0.5 * jnp.sum(((chunk(x, ids) - noisy[ids]) / noise_std)**2)
    # Rematerialize the reduction too, so backward scans retain no trace arrays.
    if chunk_checkpoint:
        chunk_loss = jax.checkpoint(chunk_loss)

    def likelihood_single(x):
        def accumulate(total, ids):
            return total + chunk_loss(x, ids), None
        # Fixed observations can be float64 even when an input is float32.
        # Scan carry dtype must match the promoted likelihood contribution.
        dtype = jnp.result_type(x, noisy, noise_std)
        total = jax.lax.scan(accumulate, jnp.zeros((), dtype=dtype), indices)[0]
        if remainder.size:
            total = total + chunk_loss(x, remainder)
        return total

    likelihood = flat_batch(likelihood_single, dim)
    logpdf = flat_batch(lambda x: logprior(x) + likelihood_single(x), dim)
    result = InverseProblem(
        logpdf, logprior, likelihood, forward, apply, inverse, truth, clean, noisy, noise_std,
        dict(n_grid=n_grid, num_sources=num_sources, num_receivers=num_receivers,
             domain_extent=25.6e-3, spacing=spacing, pml_size=pml_size,
             mask=mask, source_positions=source_positions, receiver_positions=receiver_positions,
             reference_speed=1630., cfl=0.25, t_end=25e-6, dt=time_axis.dt,
             num_time_steps=len(t), source_frequency=1e6, pulse=pulse,
             source_batch_size=source_batch_size, data_source_batch_size=data_source_batch_size,
             checkpoint_strategy=checkpoint_strategy, ell=1e-3, alpha=2., sigma_x=0.04,
             truth_seed=truth_seed, observation_seed=observation_seed),
    )
    return result if return_all else result.logpdf
