"""Adapter for the project's existing small vector autoencoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from neuripp.image_benchmarks.assets.files import sha256_path
from neuripp.image_benchmarks.encoders import project_ae_training


class ProjectAEEncoder:
    is_stochastic = False

    def __init__(
        self,
        model: Any,
        latent_dim: int,
        checkpoint: str | Path | None = None,
    ):
        if latent_dim < 1:
            raise ValueError("AE latent_dim must be positive")
        self.model = model
        self.latent_dim = int(latent_dim)
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.checkpoint_sha256 = (
            sha256_path(self.checkpoint) if self.checkpoint is not None else None
        )

    def encode(self, images: Any, rng: Any | None = None):
        del rng
        images_01 = (jnp.asarray(images) + 1.0) * 0.5
        return jax.lax.stop_gradient(jnp.asarray(self.model.encode(images_01)))

    def decode(self, latent: Any, rng: Any | None = None):
        del rng
        images_01 = self.model.decode(jnp.asarray(latent))
        return jnp.clip(images_01 * 2.0 - 1.0, -1.0, 1.0)

    def latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        del input_shape
        return (self.latent_dim,)


def load_project_ae(
    *,
    image_shape: tuple[int, ...],
    latent_dim: int,
    checkpoint: str | Path,
    train_if_missing: bool,
    train_images: np.ndarray | None = None,
    validation_images: np.ndarray | None = None,
    dataset_name: str = "image_dataset",
    training_config: dict[str, Any] | None = None,
    seed: int = 0,
    frozen_during_flow_training: bool = True,
) -> ProjectAEEncoder:
    if not frozen_during_flow_training:
        raise ValueError("The project AE must be frozen during Flow Matching training")
    module = project_ae_training
    checkpoint = Path(checkpoint).expanduser().resolve()
    model_dir = checkpoint / "model"
    metadata_path = checkpoint / "metadata.json"
    if model_dir.exists() and metadata_path.exists():
        model, _ = module.load_autoencoder_checkpoint(
            checkpoint, image_shape, latent_dim, rng_seed=seed
        )
    else:
        if not train_if_missing:
            raise FileNotFoundError(
                f"AE checkpoint is missing at {checkpoint} and train_if_missing is false"
            )
        if train_images is None or validation_images is None:
            raise ValueError(
                "train_images and validation_images are required to train a missing AE"
            )
        # The existing AE trainer uses [0, 1] image convention.
        train_01 = np.clip((np.asarray(train_images) + 1.0) * 0.5, 0.0, 1.0)
        validation_01 = np.clip(
            (np.asarray(validation_images) + 1.0) * 0.5, 0.0, 1.0
        )
        model, _ = module.train_autoencoder(
            train_images=train_01,
            test_images=validation_01,
            dataset_name=dataset_name,
            image_shape=image_shape,
            encoder_dim=latent_dim,
            checkpoint_dir=checkpoint,
            training_config=training_config,
            rng_seed=seed,
        )
    return ProjectAEEncoder(model, latent_dim, checkpoint)
