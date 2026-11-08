"""Focused checks only: pytest tests/test_pde_inverse_targets.py."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from inverse_common import fft_prior
from inverse_acoustic import (
    ACOUSTIC_CIRCLE_PHANTOM,
    ACOUSTIC_REFERENCE_SPEED,
    acoustic_reconstruction_mask,
    acoustic_sound_speed_from_log_contrast,
    acoustic_truth_fields,
    validate_positions,
)
from logpdf_targets import get_logpdf_ks_inverse_problem, get_logpdf_acoustic_inverse_problem
from benchmark_runner import field_samples_for_plot, field_truth_for_plot


@pytest.mark.parametrize("shape,spacing,ell,sigma", [
    ((24,), (100 / 24,), 2., 1.),
    ((16, 16), (0.0256 / 16,) * 2, 1e-3, 0.04),
])
def test_prior(shape, spacing, ell, sigma):
    dim = int(np.prod(shape))
    apply, inverse, logprior = fft_prior(shape, spacing, ell, sigma)
    z = jax.random.normal(jax.random.PRNGKey(8), (2, dim))
    np.testing.assert_allclose(inverse(apply(z)), z, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(logprior(apply(z)), -0.5 * jnp.sum(z*z, axis=-1), rtol=2e-5)
    # Every covariance diagonal equals sigma**2, and the DC mode survives.
    basis = apply(jnp.eye(dim))
    np.testing.assert_allclose(jnp.sum(basis*basis, axis=0), sigma**2, rtol=2e-5)
    assert np.all(np.asarray(apply(jnp.ones(dim))) > 0)


def test_positions():
    validate_positions(([2, 3], [3, 2]), 8, 1, "test")
    with pytest.raises(ValueError, match="duplicate"):
        validate_positions(([2, 2], [3, 3]), 8, 1, "test")
    with pytest.raises(ValueError, match="PML"):
        validate_positions(([0], [2]), 8, 1, "test")


@pytest.mark.parametrize("n_grid", [64, 256])
def test_documented_acoustic_circle_phantom_and_inverse_mapping(n_grid):
    jwave_geometry = pytest.importorskip("jwave.geometry")
    scale = n_grid / 256
    truth, speed = acoustic_truth_fields(n_grid)
    expected = jnp.ones((n_grid, n_grid))
    for radius, center, contrast in ACOUSTIC_CIRCLE_PHANTOM:
        expected += contrast * jwave_geometry.circ_mask(
            (n_grid, n_grid),
            radius * scale,
            (center[0] * scale, center[1] * scale),
        )
    expected *= ACOUSTIC_REFERENCE_SPEED
    np.testing.assert_allclose(speed, expected, rtol=1e-6, atol=1e-5)
    reconstructed_speed = acoustic_sound_speed_from_log_contrast(
        truth, acoustic_reconstruction_mask(n_grid)
    )
    np.testing.assert_allclose(reconstructed_speed, expected, rtol=2e-6, atol=1e-5)
    assert truth.shape == (n_grid * n_grid,)


def test_legacy_acoustic_prior_truth_is_seeded_and_distinct_from_new_default():
    pytest.importorskip("jwave")
    circle_truth, _ = acoustic_truth_fields(16)
    old_a, _ = acoustic_truth_fields(16, "prior_gaussian", truth_seed=42)
    old_b, _ = acoustic_truth_fields(16, "prior_gaussian", truth_seed=42)
    old_c, _ = acoustic_truth_fields(16, "prior_gaussian", truth_seed=43)
    np.testing.assert_array_equal(old_a, old_b)
    assert not np.array_equal(old_a, old_c)
    assert not np.array_equal(old_a, circle_truth)


def test_acoustic_plot_fields_are_physical_sound_speed():
    pytest.importorskip("jwave")
    distribution = {"name": "acoustic", "n_grid": 64}
    samples = jnp.zeros((2, 64 * 64))
    plotted_samples = field_samples_for_plot(samples, distribution)
    truth = field_truth_for_plot(distribution)
    np.testing.assert_allclose(plotted_samples, 1480.0)
    assert truth.shape == (64 * 64,)
    assert float(jnp.min(truth)) == pytest.approx(1480.0)
    assert float(jnp.max(truth)) > 1480.0
    assert float(jnp.max(truth)) < 1628.0


def test_ks_jit_gradient_data_and_reference():
    ex = pytest.importorskip("exponax")
    # Float64 gives a discriminating directional finite difference through 200 steps.
    with jax.enable_x64():
        p = get_logpdf_ks_inverse_problem(24, 5, return_all=True)
        q = get_logpdf_ks_inverse_problem(24, 5, return_all=True)
        np.testing.assert_array_equal(p.truth, q.truth)
        np.testing.assert_array_equal(p.noisy, q.noisy)
        np.testing.assert_array_equal(p.settings["sensor_indices"], [0, 4, 9, 14, 19])
        stepper = ex.stepper.KuramotoSivashinskyConservative(
            1, 100., 24, 0.1, order=2)
        reference = ex.rollout(stepper, 200, include_init=False)(p.truth[None, :])
        reference = reference[jnp.array([49, 99, 149, 199]), 0][:, p.settings["sensor_indices"]]
        np.testing.assert_allclose(p.clean, reference, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(p.noisy - p.clean,
            0.1 * jax.random.normal(jax.random.PRNGKey(43), p.clean.shape), atol=1e-15)
        x = p.truth * 0.95
        value, grad = jax.jit(jax.value_and_grad(p.logpdf))(x)
        assert np.isfinite(value) and np.all(np.isfinite(grad))
        direction = jax.random.normal(jax.random.PRNGKey(9), x.shape)
        direction /= jnp.linalg.norm(direction)
        eps = 1e-5
        fd = (p.logpdf(x + eps*direction) - p.logpdf(x - eps*direction)) / (2*eps)
        np.testing.assert_allclose(jnp.vdot(grad, direction), fd, rtol=2e-5, atol=2e-5)
        batch = jnp.stack([x, p.truth]).reshape(1, 2, 24)
        np.testing.assert_allclose(p.logpdf(batch), [[p.logpdf(x), p.logpdf(p.truth)]])
        np.testing.assert_allclose(value, p.logprior(x) + p.loglikelihood(x))
        with pytest.raises(ValueError, match="expected"):
            p.logpdf(jnp.zeros(23))


def test_ks_default_geometry():
    pytest.importorskip("exponax")
    p = get_logpdf_ks_inverse_problem(return_all=True)
    assert p.clean.shape == (4, 20)
    np.testing.assert_array_equal(p.settings["sensor_indices"], np.arange(0, 200, 10))
    assert np.isfinite(p.logpdf(p.truth))


def test_acoustic_chunking_and_gradient():
    pytest.importorskip("jwave")
    p = get_logpdf_acoustic_inverse_problem(
        32, 3, 4, source_batch_size=2, return_all=True)
    x = p.truth * 0.95
    reference = -0.5 * jnp.sum(((p.forward(x) - p.noisy) / p.noise_std)**2)
    np.testing.assert_allclose(p.loglikelihood(x), reference, rtol=2e-5)
    value, grad = jax.jit(jax.value_and_grad(p.logpdf))(x)
    assert np.isfinite(value) and np.all(np.isfinite(grad))
    for strategy in ("none", "solver", "chunk", "solver_and_chunk"):
        q = get_logpdf_acoustic_inverse_problem(
            32, 3, 4, source_batch_size=1, checkpoint_strategy=strategy,
            data_source_batch_size=3, return_all=True)
        np.testing.assert_allclose(q.clean, p.clean, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(q.logpdf(x), value, rtol=2e-5)
        other_grad = jax.jit(jax.grad(q.logpdf))(x)
        # FFT/reduction ordering varies with source batching in float32. Use
        # a relative vector error instead of relative errors near zero entries.
        assert float(jnp.linalg.norm(other_grad - grad) / jnp.linalg.norm(grad)) < 1e-4


@pytest.mark.parametrize("name", ["ks", "acoustic"])
def test_seed_independence_and_trace_reuse(name):
    pytest.importorskip("exponax" if name == "ks" else "jwave")
    getter = get_logpdf_ks_inverse_problem if name == "ks" else get_logpdf_acoustic_inverse_problem
    settings = dict(num_points=24, num_sensors=5) if name == "ks" else dict(n_grid=16, num_sources=2, num_receivers=3)
    p = getter(**settings, return_all=True)
    q = getter(**settings, observation_seed=44, return_all=True)
    if name == "acoustic":
        # The new documented phantom is deterministic and seed-independent;
        # retain a test of legacy random truth separately.
        circle_seed = getter(**settings, truth_seed=44, return_all=True)
        np.testing.assert_array_equal(p.truth, circle_seed.truth)
        r = getter(
            **settings, truth_model="prior_gaussian", truth_seed=44, return_all=True
        )
    else:
        r = getter(**settings, truth_seed=44, return_all=True)
    np.testing.assert_array_equal(p.truth, q.truth)
    np.testing.assert_array_equal(p.clean, q.clean)
    assert not np.array_equal(p.noisy, q.noisy)
    assert not np.array_equal(p.truth, r.truth)
    traces = []
    @jax.jit
    def evaluate(x):
        traces.append(True)
        return p.logpdf(x)
    evaluate(p.truth).block_until_ready()
    evaluate(p.truth * 0.99).block_until_ready()
    assert len(traces) == 1


def test_acoustic_directional_derivative():
    pytest.importorskip("jwave")
    with jax.enable_x64():
        p = get_logpdf_acoustic_inverse_problem(16, 2, 3, return_all=True)
        mixed_value, mixed_grad = jax.jit(jax.value_and_grad(p.logpdf))(p.truth.astype(jnp.float32))
        assert np.isfinite(mixed_value) and np.all(np.isfinite(mixed_grad))
        x = p.truth * 0.95
        direction = jax.random.normal(jax.random.PRNGKey(9), x.shape)
        direction /= jnp.linalg.norm(direction)
        derivative = jnp.vdot(jax.grad(p.logpdf)(x), direction)
        eps = 1e-6
        difference = (p.logpdf(x + eps * direction) - p.logpdf(x - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(derivative, difference, rtol=1e-4, atol=1e-4)
