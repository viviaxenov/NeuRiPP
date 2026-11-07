"""Focused checks only: pytest tests/test_pde_inverse_targets.py."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from inverse_common import fft_prior
from inverse_acoustic import validate_positions
from logpdf_targets import get_logpdf_ks_inverse_problem, get_logpdf_acoustic_inverse_problem


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
