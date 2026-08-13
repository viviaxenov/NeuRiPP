"""Configuration registry for benchmark representations."""

from __future__ import annotations

from typing import Any

from image_benchmarks.encoders.diffuse_vae import load_diffuse_vae
from image_benchmarks.encoders.identity import IdentityEncoder
from image_benchmarks.encoders.project_ae import load_project_ae


ENCODER_TYPES = {"none", "ae", "vae"}


def encoder_state_shape(
    config: dict[str, Any], input_shape: tuple[int, ...]
) -> tuple[int, ...]:
    encoder_type = config.get("type")
    if encoder_type == "none":
        return tuple(input_shape)
    if encoder_type == "ae":
        latent_dim = config.get("latent_dim")
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool) or latent_dim < 1:
            raise ValueError("encoder.latent_dim must be a positive integer for AE")
        return (latent_dim,)
    if encoder_type == "vae":
        if len(input_shape) != 3 or input_shape[-1] != 3:
            raise ValueError("VAE requires an (H, W, 3) input shape")
        if input_shape[0] % 8 or input_shape[1] % 8:
            raise ValueError("VAE input dimensions must be divisible by 8")
        return (input_shape[0] // 8, input_shape[1] // 8, 4)
    supported = ", ".join(sorted(ENCODER_TYPES))
    raise ValueError(f"Unknown encoder type {encoder_type!r}; expected one of: {supported}")


def build_encoder(
    config: dict[str, Any],
    input_shape: tuple[int, ...],
    *,
    seed: int,
    train_images: Any | None = None,
    validation_images: Any | None = None,
    dataset_name: str = "image_dataset",
):
    encoder_type = config.get("type")
    encoder_state_shape(config, input_shape)
    if encoder_type == "none":
        return IdentityEncoder()
    if encoder_type == "ae":
        checkpoint = config.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("encoder.checkpoint is required for AE")
        return load_project_ae(
            image_shape=input_shape,
            latent_dim=config["latent_dim"],
            checkpoint=checkpoint,
            train_if_missing=bool(config.get("train_if_missing", False)),
            train_images=train_images,
            validation_images=validation_images,
            dataset_name=dataset_name,
            training_config=config.get("training"),
            seed=seed,
            frozen_during_flow_training=bool(
                config.get("frozen_during_flow_training", True)
            ),
        )
    if config.get("implementation", "diffuse_nnx") != "diffuse_nnx":
        raise ValueError("Only encoder.implementation='diffuse_nnx' is supported for VAE")
    checkpoint = config.get("checkpoint")
    expected_sha256 = config.get("expected_sha256")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("encoder.checkpoint is required for VAE")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(
            "encoder.expected_sha256 must be a trusted 64-character checksum "
            "obtained independently from the checkpoint download. The upstream "
            "pickle must not be loaded using a trust-on-first-use checksum."
        )
    return load_diffuse_vae(
        checkpoint=checkpoint,
        auto_download=bool(config.get("auto_download", True)),
        sample_posterior=bool(config.get("sample_posterior", True)),
        seed=seed,
        expected_sha256=expected_sha256,
    )
