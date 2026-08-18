"""Diffusers-backed Stable-Diffusion VAE adapter for deterministic latent pipelines.

This is the public/reproducible VAE backend for the image benchmarks: the pinned
DiffuseNNX ``StabilityVAE`` relies on a historical ``vae_trial1.pkl`` pickle in a
private ``will-data`` GCS bucket that is not obtainable (see willisma/diffuse_nnx
issue #3). Instead, this adapter wraps ``diffusers.FlaxAutoencoderKL``
(``stabilityai/sd-vae-ft-mse``, architecture-compatible: blocks 128/256/512/512,
silu, 2 layers, latent 4, scaling factor 0.18215) and exposes the same encoder
interface the harness expects:

- ``encode_stats(images) -> (scaled_mean, scaled_std)`` (posterior statistics),
- ``encode(images, rng=None)`` (deterministic mean or posterior sample),
- ``decode(latent) -> float32 NHWC pixels in [-1, 1]``,
- ``latent_shape``, ``downsample_factor``, ``latent_channels``,
  ``scale_factor``, ``checkpoint_sha256`` (weights digest, for cache provenance),
  ``is_stochastic``.

Unlike the historical pickle adapter, latents are scaled with the model's own
``scaling_factor`` on encode and unscaled on decode, so the pair is
self-consistent (``decode(encode(x)) ~= x``).
"""

from __future__ import annotations

import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class DiffusersVAEEncoder:
    downsample_factor = 8
    latent_channels = 4

    def __init__(
        self,
        module: Any,
        params: Any,
        checkpoint_id: str,
        *,
        sample_posterior: bool = True,
    ):
        self.module = module
        self.params = jax.device_get(params)
        self.checkpoint_id = str(checkpoint_id)
        self.checkpoint_sha256 = _params_digest(self.params)
        self.sample_posterior = bool(sample_posterior)
        self.is_stochastic = self.sample_posterior
        config = getattr(module, "config", None)
        scaling = getattr(config, "scaling_factor", None)
        self.scale_factor = float(scaling) if scaling is not None else 0.18215

        # The module and scale factor are static per instance; only params and
        # images vary, so the trace is shared across cache batches and workers.
        self._encode_stats_jit = jax.jit(
            lambda params, images: _run_encode(module, params, images, self.scale_factor)
        )
        self._decode_jit = jax.jit(
            lambda params, latents: _run_decode(module, params, latents, self.scale_factor)
        )

    def latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 3 or input_shape[-1] != 3:
            raise ValueError("Diffusers VAE requires an (H, W, 3) image state")
        height, width, _ = input_shape
        if height % self.downsample_factor or width % self.downsample_factor:
            raise ValueError("VAE image dimensions must be divisible by 8")
        return (
            height // self.downsample_factor,
            width // self.downsample_factor,
            self.latent_channels,
        )

    def encode_stats(self, images: Any) -> tuple[jax.Array, jax.Array]:
        images = jnp.asarray(images, dtype=jnp.float32)
        return self._encode_stats_jit(self.params, images)

    def sample_from_stats(
        self, mean: Any, std: Any, rng: jax.Array | Any
    ) -> jax.Array:
        if hasattr(rng, "normal"):
            noise = rng.normal(jnp.shape(mean))
        else:
            noise = jax.random.normal(rng, jnp.shape(mean))
        return jnp.asarray(mean) + jnp.asarray(std) * noise

    def encode(self, images: Any, rng: Any | None = None):
        mean, std = self.encode_stats(images)
        if not self.sample_posterior:
            return mean
        if rng is None:
            raise ValueError("rng is required when sample_posterior is true")
        return self.sample_from_stats(mean, std, rng)

    def decode(self, latent: Any, rng: Any | None = None):
        del rng
        decoded = self._decode_jit(self.params, jnp.asarray(latent))
        # Diffusers flax VAE returns float pixels in [-1, 1] (NHWC).
        return jnp.clip(jnp.asarray(decoded, dtype=jnp.float32), -1.0, 1.0)


def _run_encode(
    module: Any, params: Any, images: jax.Array, scale_factor: float
) -> tuple[jax.Array, jax.Array]:
    # Diffusers flax AutoencoderKL: NCHW images in, posterior NHWC out.
    result = module.apply(
        {"params": params}, jnp.transpose(images, (0, 3, 1, 2)), method=module.encode
    )
    posterior = result.latent_dist if hasattr(result, "latent_dist") else result
    return posterior.mean * scale_factor, posterior.std * scale_factor


def _run_decode(
    module: Any, params: Any, latents: jax.Array, scale_factor: float
) -> jax.Array:
    if latents.shape[-1] != module.config.latent_channels:
        # NCHW latents provided; the decoder consumes NHWC.
        latents = jnp.transpose(latents, (0, 2, 3, 1))
    result = module.apply(
        {"params": params}, latents / scale_factor, method=module.decode
    )
    if hasattr(result, "sample"):
        result = result.sample
    # Decoder returns NCHW pixels; convert back to the NHWC harness convention.
    return jnp.transpose(result, (0, 2, 3, 1))


def _params_digest(params: Any) -> str:
    """Stable digest over the loaded VAE weights (sorted leaf paths)."""
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    flat.sort(key=lambda item: "/".join(str(part) for part in item[0]))
    hasher = hashlib.sha256()
    for _, leaf in flat:
        array = np.asarray(leaf)
        hasher.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def load_diffusers_vae(
    *,
    checkpoint_id: str,
    sample_posterior: bool,
    seed: int = 0,
) -> DiffusersVAEEncoder:
    """Load the public SD-VAE through diffusers (lazy dependency)."""
    del seed
    from diffusers import FlaxAutoencoderKL

    result = FlaxAutoencoderKL.from_pretrained(
        checkpoint_id, from_pt=True, dtype=jnp.float32
    )
    if isinstance(result, tuple):
        module, params = result
    else:
        module = result
        params = module.params
    return DiffusersVAEEncoder(
        module, params, checkpoint_id, sample_posterior=sample_posterior
    )