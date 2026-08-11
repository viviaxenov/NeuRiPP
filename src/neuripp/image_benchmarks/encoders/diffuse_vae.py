"""Adapter for the pinned DiffuseNNX Stable-Diffusion VAE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from neuripp.image_benchmarks.assets.diffuse_nnx import (
    import_diffuse_module,
    prepare_diffuse_nnx_source,
)
from neuripp.image_benchmarks.assets.files import prepare_vae_checkpoint, sha256_path


class DiffuseVAEEncoder:
    downsample_factor = 8
    latent_channels = 4
    scale_factor = 0.18215

    def __init__(
        self,
        model: Any,
        checkpoint: str | Path,
        *,
        sample_posterior: bool = True,
    ):
        self.model = model
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.checkpoint_sha256 = sha256_path(self.checkpoint)
        self.sample_posterior = bool(sample_posterior)
        self.is_stochastic = self.sample_posterior

    def latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(input_shape) != 3 or input_shape[-1] != 3:
            raise ValueError("Diffuse VAE requires an (H, W, 3) image state")
        height, width, _ = input_shape
        if height % self.downsample_factor or width % self.downsample_factor:
            raise ValueError("VAE image dimensions must be divisible by 8")
        return (
            height // self.downsample_factor,
            width // self.downsample_factor,
            self.latent_channels,
        )

    def encode_stats(self, images: Any) -> tuple[jax.Array, jax.Array]:
        images = jnp.asarray(images)
        hidden = self.model.vae.encoder(images, deterministic=True)
        moments = self.model.vae.quant_conv(hidden)
        mean, log_variance = jnp.split(moments, 2, axis=-1)
        log_variance = jnp.clip(log_variance, -30.0, 20.0)
        std = jnp.exp(0.5 * log_variance)
        return mean * self.scale_factor, std * self.scale_factor

    def sample_from_stats(
        self, mean: Any, std: Any, rng: jax.Array | nnx.Rngs
    ) -> jax.Array:
        if isinstance(rng, nnx.Rngs):
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
        decoded = self.model.decode(jnp.asarray(latent), deterministic=True)
        if jnp.issubdtype(decoded.dtype, jnp.integer):
            return decoded.astype(jnp.float32) / 127.5 - 1.0
        decoded = decoded.astype(jnp.float32)
        # The upstream wrapper returns uint8, but injected/test implementations
        # may expose model-space images directly.
        return jnp.clip(decoded, -1.0, 1.0)


def load_diffuse_vae(
    *,
    checkpoint: str | Path,
    source_dir: str | Path,
    auto_download: bool,
    source_auto_download: bool = True,
    sample_posterior: bool,
    seed: int = 0,
    expected_sha256: str | None = None,
) -> DiffuseVAEEncoder:
    if not expected_sha256:
        raise ValueError(
            "expected_sha256 is required before loading the pickle-based VAE checkpoint. "
            "Prepare the asset first, record its checksum, then configure that checksum."
        )
    metadata = prepare_vae_checkpoint(
        checkpoint,
        auto_download=auto_download,
        expected_sha256=expected_sha256,
    )
    source_dir = prepare_diffuse_nnx_source(
        source_dir, auto_download=source_auto_download
    )
    module = import_diffuse_module("networks.encoders.sd_vae", source_dir)
    try:
        import ml_collections
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "DiffuseNNX VAE requires the 'image-benchmarks' optional dependencies"
        ) from error
    model = module.StabilityVAE(
        ml_collections.ConfigDict(),
        pretrained_path=metadata["path"],
        encoded_pixels=False,
        rngs=nnx.Rngs(seed, gaussian=seed + 1),
    )
    return DiffuseVAEEncoder(
        model, metadata["path"], sample_posterior=sample_posterior
    )
