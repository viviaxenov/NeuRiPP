"""Representation reconstruction diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def reconstruction_metrics(
    encoder,
    images,
    *,
    seed: int = 0,
) -> dict[str, float]:
    images = jnp.asarray(images, dtype=jnp.float32)
    if hasattr(encoder, "encode_stats"):
        latent = encoder.encode_stats(images)[0]
    else:
        try:
            latent = encoder.encode(images, jax.random.key(seed))
        except TypeError:
            latent = encoder.encode(images)
    reconstruction = jnp.asarray(encoder.decode(latent), dtype=jnp.float32)
    if reconstruction.shape != images.shape:
        raise ValueError(
            f"Reconstruction shape {reconstruction.shape} does not match {images.shape}"
        )
    mse = float(jnp.mean((reconstruction - images) ** 2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * jnp.log10(4.0 / mse))
    return {
        "encoder_recon_mse": mse,
        "encoder_recon_psnr": psnr,
    }
