from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import trange

from rhs_architectures import AutoEncoder


def default_encoder_training_config() -> dict[str, Any]:
    return {
        "optimizer": "adam",
        "epochs": 600,
        "learning_rate": 1e-3,
        "lr_drop_every": 150,
        "lr_drop_factor": 0.1,
        "batch_size": 1024,
        "eval_batch_size": 2048,
        "plot_every": 20,
    }


def merged_encoder_training_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = default_encoder_training_config()
    if config:
        merged.update(config)
    return merged


def encoder_checkpoint_dir(
    encoder_path: str | Path, dataset_name: str, encoder_dim: int
) -> Path:
    return (Path(encoder_path) / dataset_name / f"dim{encoder_dim}").resolve()


def build_autoencoder(
    image_shape: tuple[int, ...], encoder_dim: int, rng_seed: int = 0
) -> AutoEncoder:
    return AutoEncoder(image_shape, encoder_dim, rngs=nnx.Rngs(rng_seed))


def _checkpoint_paths(checkpoint_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        checkpoint_dir / "model",
        checkpoint_dir / "metadata.json",
        checkpoint_dir / "loss_history.npz",
        checkpoint_dir / "training_losses.pdf",
        checkpoint_dir / "reconstruction_examples.pdf",
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_loss_history(
    output_path: Path,
    eval_epochs: list[int],
    train_losses: list[float],
    test_losses: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(eval_epochs, train_losses, label="Train MSE")
    ax.plot(eval_epochs, test_losses, label="Test MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_reconstruction_examples(
    output_path: Path,
    originals: np.ndarray | jax.Array,
    reconstructions: np.ndarray | jax.Array,
) -> None:
    originals = np.asarray(originals)
    reconstructions = np.asarray(reconstructions)
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    subfigures = fig.subfigures(1, 2, wspace=0.02)

    for subfigure, images, title in zip(
        subfigures,
        (reconstructions, originals),
        ("Reconstructions", "Test images"),
        strict=True,
    ):
        axes = subfigure.subplots(6, 6)
        subfigure.suptitle(title)
        for ax, image in zip(axes.flat, images, strict=True):
            ax.imshow(np.squeeze(image), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")

    fig.savefig(output_path)
    plt.close(fig)


def _batched_apply(fn, data: jax.Array, batch_size: int) -> jax.Array:
    outputs = []
    for start in range(0, int(data.shape[0]), batch_size):
        outputs.append(fn(data[start : start + batch_size]))
    return jnp.concatenate(outputs, axis=0)


def load_autoencoder_checkpoint(
    checkpoint_dir: str | Path,
    image_shape: tuple[int, ...],
    encoder_dim: int,
    rng_seed: int = 0,
) -> tuple[AutoEncoder, dict[str, Any]]:
    model_dir, metadata_path, _, _, _ = _checkpoint_paths(Path(checkpoint_dir))
    autoencoder = build_autoencoder(image_shape, encoder_dim, rng_seed=rng_seed)
    graphdef, state = nnx.split(autoencoder)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(model_dir, target=state)
    return nnx.merge(graphdef, state), _load_json(metadata_path)


def train_autoencoder(
    train_images: np.ndarray,
    test_images: np.ndarray,
    dataset_name: str,
    image_shape: tuple[int, ...],
    encoder_dim: int,
    checkpoint_dir: str | Path,
    training_config: dict[str, Any] | None = None,
    rng_seed: int = 0,
) -> tuple[AutoEncoder, dict[str, Any]]:
    config = merged_encoder_training_config(training_config)
    optimizer_name = config["optimizer"]
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported encoder_training.optimizer {optimizer_name!r}; expected 'adam'")

    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir, metadata_path, history_path, plot_path, reconstruction_plot_path = (
        _checkpoint_paths(checkpoint_dir)
    )

    autoencoder = build_autoencoder(image_shape, encoder_dim, rng_seed=rng_seed)
    train_data = jnp.asarray(train_images, dtype=jnp.float32)
    test_data = jnp.asarray(test_images, dtype=jnp.float32)

    @nnx.jit
    def train_step(model: AutoEncoder, optimizer: nnx.Optimizer, batch: jax.Array):
        def loss_fn(current_model):
            reconstruction = current_model(batch)
            return jnp.mean((reconstruction - batch) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_batch(model: AutoEncoder, batch: jax.Array) -> jax.Array:
        reconstruction = model(batch)
        return jnp.mean((reconstruction - batch) ** 2)

    batch_size = int(config["batch_size"])
    eval_batch_size = int(config["eval_batch_size"])
    plot_every = int(config["plot_every"])
    epochs = int(config["epochs"])
    steps_per_epoch = math.ceil(int(train_data.shape[0]) / batch_size)

    schedule = optax.exponential_decay(
        init_value=config["learning_rate"],
        transition_steps=steps_per_epoch * config["lr_drop_every"],
        decay_rate=config["lr_drop_factor"],
        staircase=True,
    )
    optimizer = nnx.Optimizer(
        autoencoder,
        optax.adam(schedule),
        wrt=nnx.Param,
    )

    eval_epochs: list[int] = []
    train_losses: list[float] = []
    test_losses: list[float] = []
    best_train = float("inf")
    rng = np.random.default_rng(rng_seed)
    plot_rng = np.random.default_rng(rng_seed)
    plot_indices = plot_rng.choice(
        int(test_data.shape[0]), size=36, replace=int(test_data.shape[0]) < 36
    )
    plot_images = test_data[plot_indices]
    pbar = trange(epochs, desc=f"AE {dataset_name} dim{encoder_dim}")

    for epoch in pbar:
        permutation = rng.permutation(train_data.shape[0])
        shuffled = train_data[permutation]
        batch_losses = []
        for start in range(0, int(shuffled.shape[0]), batch_size):
            batch = shuffled[start : start + batch_size]
            batch_losses.append(float(train_step(autoencoder, optimizer, batch)))

        epoch_train_loss = float(np.mean(batch_losses))
        best_train = min(best_train, epoch_train_loss)
        current_lr = float(schedule((epoch + 1) * steps_per_epoch))
        pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.6f}",
            best_train=f"{best_train:.6f}",
            lr=f"{current_lr:.2e}",
        )

        if (epoch + 1) % plot_every == 0 or epoch == epochs - 1:
            test_batch_losses = []
            for start in range(0, int(test_data.shape[0]), eval_batch_size):
                batch = test_data[start : start + eval_batch_size]
                test_batch_losses.append(float(eval_batch(autoencoder, batch)))
            eval_epochs.append(epoch + 1)
            train_losses.append(epoch_train_loss)
            test_losses.append(float(np.mean(test_batch_losses)))
            _plot_loss_history(plot_path, eval_epochs, train_losses, test_losses)
            reconstructions = autoencoder.decode(autoencoder.encode(plot_images))
            _plot_reconstruction_examples(
                reconstruction_plot_path, plot_images, reconstructions
            )

    graphdef, state = nnx.split(autoencoder)
    checkpointer = ocp.StandardCheckpointer()
    if model_dir.exists():
        import shutil

        shutil.rmtree(model_dir)
    checkpointer.save(model_dir, state)
    checkpointer.wait_until_finished()

    metadata = {
        "dataset_name": dataset_name,
        "encoder_dim": encoder_dim,
        "image_shape": list(image_shape),
        "training_config": config,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_test_loss": test_losses[-1] if test_losses else None,
        "best_test_loss": min(test_losses) if test_losses else None,
        "best_test_epoch": eval_epochs[int(np.argmin(test_losses))] if test_losses else None,
    }
    _write_json(metadata_path, metadata)
    np.savez(
        history_path,
        eval_epochs=np.asarray(eval_epochs, dtype=int),
        train_loss=np.asarray(train_losses, dtype=float),
        test_loss=np.asarray(test_losses, dtype=float),
    )
    return nnx.merge(graphdef, state), metadata


def load_or_train_autoencoder(
    train_images: np.ndarray,
    test_images: np.ndarray,
    dataset_name: str,
    image_shape: tuple[int, ...],
    encoder_dim: int,
    encoder_path: str | Path = "./encoder",
    training_config: dict[str, Any] | None = None,
    rng_seed: int = 0,
) -> tuple[AutoEncoder, dict[str, Any], Path]:
    checkpoint_dir = encoder_checkpoint_dir(encoder_path, dataset_name, encoder_dim)
    model_dir, metadata_path, _, _, _ = _checkpoint_paths(checkpoint_dir)
    if model_dir.exists() and metadata_path.exists():
        autoencoder, metadata = load_autoencoder_checkpoint(
            checkpoint_dir,
            image_shape,
            encoder_dim,
            rng_seed=rng_seed,
        )
        return autoencoder, metadata, checkpoint_dir

    autoencoder, metadata = train_autoencoder(
        train_images=train_images,
        test_images=test_images,
        dataset_name=dataset_name,
        image_shape=image_shape,
        encoder_dim=encoder_dim,
        checkpoint_dir=checkpoint_dir,
        training_config=training_config,
        rng_seed=rng_seed,
    )
    return autoencoder, metadata, checkpoint_dir


def encode_dataset(
    autoencoder: AutoEncoder, images: np.ndarray | jax.Array, batch_size: int = 2048
) -> jax.Array:
    data = jnp.asarray(images, dtype=jnp.float32)
    return _batched_apply(autoencoder.encode, data, batch_size=batch_size)


def normalize_latents(
    train_latents: jax.Array, test_latents: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    latent_mean = jnp.mean(train_latents, axis=0)
    latent_std = jnp.std(train_latents, axis=0)
    latent_std = jnp.maximum(latent_std, 1e-20)
    train_norm = (train_latents - latent_mean[None, :]) / latent_std[None, :]
    test_norm = (test_latents - latent_mean[None, :]) / latent_std[None, :]
    return train_norm, test_norm, latent_mean, latent_std


def prepare_encoded_image_dataset(
    dataset_name: str,
    train_images: np.ndarray,
    test_images: np.ndarray,
    encoder_dim: int,
    encoder_path: str | Path = "./encoder",
    training_config: dict[str, Any] | None = None,
    rng_seed: int = 0,
) -> dict[str, Any]:
    image_shape = tuple(train_images.shape[1:])
    autoencoder, metadata, checkpoint_dir = load_or_train_autoencoder(
        train_images=train_images,
        test_images=test_images,
        dataset_name=dataset_name,
        image_shape=image_shape,
        encoder_dim=encoder_dim,
        encoder_path=encoder_path,
        training_config=training_config,
        rng_seed=rng_seed,
    )
    train_latents = encode_dataset(autoencoder, train_images)
    test_latents = encode_dataset(autoencoder, test_images)
    train_norm, test_norm, latent_mean, latent_std = normalize_latents(
        train_latents, test_latents
    )
    latent_log_det = float(jnp.log(latent_std).sum())
    return {
        "train": np.asarray(train_norm, dtype=np.float32),
        "test": np.asarray(test_norm, dtype=np.float32),
        "latent_mean": np.asarray(latent_mean, dtype=np.float32),
        "latent_std": np.asarray(latent_std, dtype=np.float32),
        "image_shape": image_shape,
        "eval_mode": "latent",
        "latent_log_det_per_example": latent_log_det,
        "encoder_dim": encoder_dim,
        "autoencoder": autoencoder,
        "autoencoder_checkpoint_dir": checkpoint_dir,
        "autoencoder_metadata": metadata,
    }


def decode_latents_to_images(latents_normalized: jax.Array, context: dict[str, Any]) -> jax.Array:
    autoencoder = context.get("autoencoder")
    if autoencoder is None:
        autoencoder, _ = load_autoencoder_checkpoint(
            context["autoencoder_checkpoint_dir"],
            tuple(context["image_shape"]),
            int(context["encoder_dim"]),
        )
    latent_mean = jnp.asarray(context["latent_mean"])
    latent_std = jnp.asarray(context["latent_std"])
    latents = jnp.asarray(latents_normalized) * latent_std[None, :] + latent_mean[None, :]
    return jnp.clip(autoencoder.decode(latents), 0.0, 1.0)
