"""j-Wave raw-trace acoustic inverse benchmark with bounded source memory."""

import numpy as np
import jax
import jax.numpy as jnp

if __package__:
    from .inverse_common import InverseProblem, fft_prior, flat_batch, positive_int
else:
    from inverse_common import InverseProblem, fft_prior, flat_batch, positive_int


ACOUSTIC_REFERENCE_SPEED = 1480.0
ACOUSTIC_CIRCLE_PHANTOM = (
    (16.0, (100.0, 100.0), 0.05),
    (20.0, (165.0, 128.0), 0.02),
    (40.0, (130.0, 130.0), 0.03),
)


def acoustic_reconstruction_mask(n_grid):
    """Return the scaled outer reconstruction circle used by the speed map."""

    from jwave.geometry import circ_mask

    scale = n_grid / 256
    return jnp.asarray(
        circ_mask((n_grid, n_grid), 80 * scale, (n_grid / 2, n_grid / 2))
    )


def acoustic_sound_speed_from_log_contrast(x, mask):
    """Map flat (..., n_grid**2) log-contrast fields to sound speed in m/s."""

    x = jnp.asarray(x)
    mask = jnp.asarray(mask)
    n_grid = mask.shape[0]
    if mask.shape != (n_grid, n_grid) or x.shape[-1] != n_grid * n_grid:
        raise ValueError("log-contrast shape must match the square acoustic mask")
    field = x.reshape((*x.shape[:-1], n_grid, n_grid))
    return ACOUSTIC_REFERENCE_SPEED * (
        1.0 + 0.10 * mask * jnp.tanh(field / 0.10)
    )


def acoustic_truth_fields(
    n_grid,
    truth_model="jwave_fwi_three_circles",
    truth_seed=None,
    mask=None,
):
    """Return flat truth ``x`` and its matching physical sound-speed field.

    The documented j-Wave FWI phantom is defined in sound-speed space. Its
    corresponding log-contrast is obtained by inverting this benchmark's
    bounded speed parameterization. ``prior_gaussian`` is retained only for
    interpreting checkpoints made before the documented phantom was adopted.
    """

    n_grid = positive_int("n_grid", n_grid)
    if truth_model not in {"jwave_fwi_three_circles", "prior_gaussian"}:
        raise ValueError(f"Unsupported acoustic truth_model {truth_model!r}")
    if mask is None:
        mask = acoustic_reconstruction_mask(n_grid)
    else:
        mask = jnp.asarray(mask)
    if mask.shape != (n_grid, n_grid):
        raise ValueError("acoustic mask must match n_grid")

    if truth_model == "jwave_fwi_three_circles":
        from jwave.geometry import circ_mask

        scale = n_grid / 256
        relative_speed = jnp.ones((n_grid, n_grid))
        for radius, center, contrast in ACOUSTIC_CIRCLE_PHANTOM:
            circle = circ_mask(
                (n_grid, n_grid),
                radius * scale,
                (center[0] * scale, center[1] * scale),
            )
            relative_speed = relative_speed + contrast * jnp.asarray(circle)
        speed = ACOUSTIC_REFERENCE_SPEED * relative_speed
        safe_mask = jnp.where(mask > 0, mask, 1.0)
        normalized_contrast = jnp.where(
            mask > 0,
            (speed / ACOUSTIC_REFERENCE_SPEED - 1.0) / (0.10 * safe_mask),
            0.0,
        )
        if bool(jnp.any(jnp.abs(normalized_contrast) >= 1.0)):
            raise ValueError("circle phantom exceeds the bounded speed transform")
        truth = (0.10 * jnp.arctanh(normalized_contrast)).reshape((-1,))
    else:
        if truth_seed is None:
            truth_seed = 42
        if isinstance(truth_seed, bool) or not isinstance(truth_seed, int) or truth_seed < 0:
            raise ValueError("truth_seed must be a nonnegative integer for prior_gaussian")
        spacing = 25.6e-3 / n_grid
        apply, _, _ = fft_prior((n_grid, n_grid), (spacing, spacing), 1e-3, 0.04)
        truth = apply(
            jax.random.normal(jax.random.PRNGKey(truth_seed), (n_grid * n_grid,))
        )

    sound_speed = acoustic_sound_speed_from_log_contrast(truth, mask)
    return truth, sound_speed


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
    n_grid=256, num_sources=64, num_receivers=64, truth_seed=None,
    observation_seed=43, source_batch_size=1,
    checkpoint_strategy="solver_and_chunk", data_source_batch_size=None,
    return_all=False, truth_model="jwave_fwi_three_circles",
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
    from jwave.geometry import Domain, Medium, Sources, Sensors, TimeAxis, points_on_circle
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
    mask = acoustic_reconstruction_mask(n_grid)
    time_axis = TimeAxis.from_medium(
        Medium(domain, sound_speed=1630.0), cfl=0.25, t_end=25e-6)
    t = time_axis.to_array()
    pulse = (1.3e-5 / time_axis.dt) * jnp.sin(2 * jnp.pi * 1e6 * t)
    pulse = gaussian_window(apply_ramp(pulse, time_axis.dt, 1e6), t, 3e-6, 1.2e-6)
    solver_checkpoint = checkpoint_strategy in ("solver", "solver_and_chunk")
    chunk_checkpoint = checkpoint_strategy in ("chunk", "solver_and_chunk")
    solver_settings = TimeWavePropagationSettings(checkpoint=solver_checkpoint)

    def single_source(x, index):
        speed = acoustic_sound_speed_from_log_contrast(x, mask)
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
    truth, sound_speed_truth = acoustic_truth_fields(
        n_grid, truth_model=truth_model, truth_seed=truth_seed, mask=mask
    )
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
    target_settings = dict(
        n_grid=n_grid,
        num_sources=num_sources,
        num_receivers=num_receivers,
        domain_extent=25.6e-3,
        spacing=spacing,
        pml_size=pml_size,
        mask=mask,
        source_positions=source_positions,
        receiver_positions=receiver_positions,
        reference_speed=1630.0,
        cfl=0.25,
        t_end=25e-6,
        dt=time_axis.dt,
        num_time_steps=len(t),
        source_frequency=1e6,
        pulse=pulse,
        source_batch_size=source_batch_size,
        data_source_batch_size=data_source_batch_size,
        checkpoint_strategy=checkpoint_strategy,
        ell=1e-3,
        alpha=2.0,
        sigma_x=0.04,
        truth_model=truth_model,
        observation_seed=observation_seed,
        sound_speed_truth=sound_speed_truth,
    )
    if truth_model == "prior_gaussian":
        target_settings["truth_seed"] = 42 if truth_seed is None else truth_seed
    result = InverseProblem(
        logpdf, logprior, likelihood, forward, apply, inverse, truth, clean, noisy, noise_std,
        target_settings,
    )
    return result if return_all else result.logpdf
