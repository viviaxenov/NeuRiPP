"""Shared flat-parameter interfaces for the PDE inverse examples."""

from dataclasses import dataclass
from typing import Callable, Mapping, Any
import math
import numbers

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class InverseProblem:
    """Fixed synthetic inverse problem (densities omit additive constants).

    All callables accept physical parameters of shape ``(..., dim)``; leading
    batch axes are vectorized with ``jax.vmap``. Densities return the leading
    shape, forward returns that shape plus the observation shape, and prior
    operators preserve the input shape. ``truth`` is flat;
    ``clean`` and ``noisy`` have the native observation shape. ``noise_std``
    is the fixed scalar observation standard deviation. ``settings`` records
    solver, geometry, prior and seed choices. The prior is full rank: its
    constant Fourier mode is retained, not removed realization by realization.
    """

    logpdf: Callable
    logprior: Callable
    loglikelihood: Callable
    forward: Callable
    prior_apply: Callable
    prior_inverse: Callable
    truth: jax.Array
    clean: jax.Array
    noisy: jax.Array
    noise_std: jax.Array
    settings: Mapping[str, Any]


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def flat_batch(single, dim):
    """Vectorize a single flat-vector function over any leading batch axes."""
    def evaluate(x):
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != dim:
            raise ValueError(f"expected (..., {dim}), got {x.shape}")
        if x.ndim == 1:
            return single(x)
        result = jax.vmap(single)(x.reshape((-1, dim)))
        return result.reshape(x.shape[:-1] + result.shape[1:])
    return jax.jit(evaluate)


def fft_prior(shape, spacing, ell, sigma, alpha=2.0):
    """Return apply/inverse/logdensity for the normalized FFT Gaussian prior.

    ``spacing`` and ``ell`` use the same physical units. The RMS-normalized
    positive multiplier gives marginal variance sigma**2 at every grid point.
    """
    frequencies = jnp.meshgrid(
        *(2 * jnp.pi * jnp.fft.fftfreq(n, d=d) for n, d in zip(shape, spacing)),
        indexing="ij",
    )
    h = (1 + ell**2 * sum(w**2 for w in frequencies)) ** (-alpha / 2)
    multiplier = sigma * h / jnp.sqrt(jnp.mean(h**2))
    dim = math.prod(shape)

    def transform(x, inverse=False):
        spectrum = jnp.fft.fftn(x.reshape(shape))
        spectrum = spectrum / multiplier if inverse else spectrum * multiplier
        return jnp.fft.ifftn(spectrum).real.reshape((dim,))

    apply = flat_batch(transform, dim)
    inverse = flat_batch(lambda x: transform(x, True), dim)
    logprior = flat_batch(lambda x: -0.5 * jnp.sum(transform(x, True)**2), dim)
    return apply, inverse, logprior
