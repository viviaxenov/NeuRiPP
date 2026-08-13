"""Checkpoint and training entry points for the packaged project AE."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from image_benchmarks.encoders.project_ae_model import AutoEncoder


def build_autoencoder(
    image_shape: tuple[int, ...], latent_dim: int, rng_seed: int = 0
) -> AutoEncoder:
    return AutoEncoder(image_shape, latent_dim, nnx.Rngs(rng_seed))


def load_autoencoder_checkpoint(
    checkpoint_dir: str | Path,
    image_shape: tuple[int, ...],
    encoder_dim: int,
    rng_seed: int = 0,
) -> tuple[AutoEncoder, dict[str, Any]]:
    import orbax.checkpoint as ocp

    checkpoint_dir = Path(checkpoint_dir)
    model = build_autoencoder(image_shape, encoder_dim, rng_seed)
    graph, state = nnx.split(model)
    restored = ocp.StandardCheckpointer().restore(checkpoint_dir / "model", target=state)
    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    return nnx.merge(graph, restored), metadata


def train_autoencoder(
    *,
    train_images: np.ndarray,
    test_images: np.ndarray,
    dataset_name: str,
    image_shape: tuple[int, ...],
    encoder_dim: int,
    checkpoint_dir: str | Path,
    training_config: dict[str, Any] | None = None,
    rng_seed: int = 0,
) -> tuple[AutoEncoder, dict[str, Any]]:
    """Train the existing small AE without benchmark-run plotting side effects."""

    import orbax.checkpoint as ocp

    config = {
        "epochs": 600,
        "learning_rate": 1e-3,
        "lr_drop_every": 150,
        "lr_drop_factor": 0.1,
        "batch_size": 1024,
        "eval_batch_size": 2048,
    }
    config.update(training_config or {})
    model = build_autoencoder(image_shape, encoder_dim, rng_seed)
    train = jnp.asarray(train_images, dtype=jnp.float32)
    test = jnp.asarray(test_images, dtype=jnp.float32)
    steps_per_epoch = math.ceil(len(train) / int(config["batch_size"]))
    schedule = optax.exponential_decay(
        config["learning_rate"],
        steps_per_epoch * int(config["lr_drop_every"]),
        config["lr_drop_factor"],
        staircase=True,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    @nnx.jit
    def train_step(current_model, current_optimizer, batch):
        def loss_fn(candidate):
            return jnp.mean((candidate(batch) - batch) ** 2)

        loss, gradients = nnx.value_and_grad(loss_fn)(current_model)
        current_optimizer.update(current_model, gradients)
        return loss

    rng = np.random.default_rng(rng_seed)
    final_train_loss = None
    for _ in range(int(config["epochs"])):
        permutation = rng.permutation(len(train))
        losses = []
        for start in range(0, len(train), int(config["batch_size"])):
            batch = train[permutation[start : start + int(config["batch_size"])]]
            losses.append(float(train_step(model, optimizer, batch)))
        final_train_loss = float(np.mean(losses))

    test_losses = []
    for start in range(0, len(test), int(config["eval_batch_size"])):
        batch = test[start : start + int(config["eval_batch_size"])]
        test_losses.append(float(jnp.mean((model(batch) - batch) ** 2)))
    test_loss = float(np.mean(test_losses))

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model"
    if model_path.exists():
        shutil.rmtree(model_path)
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(model_path, state)
    checkpointer.wait_until_finished()
    metadata = {
        "dataset_name": dataset_name,
        "encoder_dim": encoder_dim,
        "image_shape": list(image_shape),
        "training_config": config,
        "final_train_loss": final_train_loss,
        "final_test_loss": test_loss,
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model, metadata
