from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
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


@dataclass
class LoadedAutoencoder:
    """Autoencoder and training-latent statistics loaded from one checkpoint."""

    autoencoder: AutoEncoder
    latent_mean: jax.Array
    latent_std: jax.Array
    latent_covariance: jax.Array
    metadata: dict[str, Any]
    checkpoint_dir: Path


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


def _latent_statistics_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "latent_statistics.npz"


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


def plot_image_grid(
    images: np.ndarray | jax.Array,
    nrows: int,
    ncols: int,
    *,
    title: str | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Plot the first ``nrows * ncols`` images and return the figure and axes."""

    if isinstance(nrows, bool) or not isinstance(nrows, int) or nrows < 1:
        raise ValueError("nrows must be a positive integer")
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer")

    images = np.asarray(images)
    image_count = nrows * ncols
    if images.ndim < 3 or images.shape[0] < image_count:
        raise ValueError(
            f"Expected at least {image_count} images, got shape {images.shape}"
        )

    if figsize is None:
        figsize = (1.8 * ncols, 1.8 * nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        layout="constrained",
    )
    for ax, image in zip(axes.flat, images[:image_count], strict=True):
        ax.imshow(np.squeeze(image), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_axis_off()
    if title is not None:
        fig.suptitle(title)
    return fig, axes


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


def _compute_latent_statistics(
    train_latents: np.ndarray | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    train_latents = jnp.asarray(train_latents, dtype=jnp.float32)
    if train_latents.ndim != 2 or train_latents.shape[0] < 1:
        raise ValueError("Training latents must be a non-empty rank-2 array")
    latent_mean = jnp.mean(train_latents, axis=0)
    latent_std = jnp.maximum(jnp.std(train_latents, axis=0), 1e-20)
    centered = train_latents - latent_mean[None, :]
    latent_covariance = centered.T @ centered / train_latents.shape[0]
    return latent_mean, latent_std, latent_covariance


def _write_latent_statistics(
    checkpoint_dir: Path,
    latent_mean: np.ndarray | jax.Array,
    latent_std: np.ndarray | jax.Array,
    latent_covariance: np.ndarray | jax.Array,
) -> None:
    output_path = _latent_statistics_path(checkpoint_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".npz",
            delete=False,
        ) as f:
            temporary_path = Path(f.name)
            np.savez(
                f,
                mean=np.asarray(latent_mean, dtype=np.float32),
                std=np.asarray(latent_std, dtype=np.float32),
                covariance=np.asarray(latent_covariance, dtype=np.float32),
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_latent_statistics(
    checkpoint_dir: Path, encoder_dim: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    path = _latent_statistics_path(checkpoint_dir)
    with np.load(path) as statistics:
        latent_mean = np.asarray(statistics["mean"], dtype=np.float32)
        latent_std = np.asarray(statistics["std"], dtype=np.float32)
        latent_covariance = np.asarray(statistics["covariance"], dtype=np.float32)

    expected_vector_shape = (encoder_dim,)
    expected_covariance_shape = (encoder_dim, encoder_dim)
    if latent_mean.shape != expected_vector_shape:
        raise ValueError(
            f"Invalid latent mean shape in {path}: expected {expected_vector_shape}, "
            f"got {latent_mean.shape}"
        )
    if latent_std.shape != expected_vector_shape:
        raise ValueError(
            f"Invalid latent std shape in {path}: expected {expected_vector_shape}, "
            f"got {latent_std.shape}"
        )
    if latent_covariance.shape != expected_covariance_shape:
        raise ValueError(
            f"Invalid latent covariance shape in {path}: expected "
            f"{expected_covariance_shape}, got {latent_covariance.shape}"
        )
    return (
        jnp.asarray(latent_mean),
        jnp.asarray(latent_std),
        jnp.asarray(latent_covariance),
    )


def _load_training_images(dataset_name: str) -> np.ndarray:
    from datasets import load_dataset

    dataset_ids = {
        "mnist": "ylecun/mnist",
        "fashion_mnist": "zalando-datasets/fashion_mnist",
    }
    try:
        dataset_id = dataset_ids[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(dataset_ids))
        raise ValueError(
            f"Cannot migrate autoencoder statistics for dataset {dataset_name!r}; "
            f"provide train_images or use one of: {supported}"
        ) from exc
    dataset = load_dataset(dataset_id, split="train")
    return np.stack(
        [np.asarray(image, dtype=np.float32) for image in dataset["image"]], axis=0
    ) / 255.0


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


def load_autoencoder(
    checkpoint_dir: str | Path,
    *,
    train_images: np.ndarray | jax.Array | None = None,
    batch_size: int = 2048,
    rng_seed: int = 0,
) -> LoadedAutoencoder:
    """Load an autoencoder bundle, migrating legacy checkpoints when needed."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    try:
        image_shape = tuple(int(size) for size in metadata["image_shape"])
        encoder_dim = int(metadata["encoder_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid autoencoder metadata in {metadata_path}: image_shape and "
            "encoder_dim are required"
        ) from exc

    autoencoder, metadata = load_autoencoder_checkpoint(
        checkpoint_dir,
        image_shape,
        encoder_dim,
        rng_seed=rng_seed,
    )
    statistics_path = _latent_statistics_path(checkpoint_dir)
    if not statistics_path.exists():
        if train_images is None:
            dataset_name = metadata.get("dataset_name")
            if not isinstance(dataset_name, str):
                raise ValueError(
                    f"Cannot migrate {checkpoint_dir}: metadata.json has no dataset_name; "
                    "provide train_images"
                )
            train_images = _load_training_images(dataset_name)
        train_latents = encode_dataset(autoencoder, train_images, batch_size=batch_size)
        latent_mean, latent_std, latent_covariance = _compute_latent_statistics(
            train_latents
        )
        _write_latent_statistics(
            checkpoint_dir, latent_mean, latent_std, latent_covariance
        )

    latent_mean, latent_std, latent_covariance = _load_latent_statistics(
        checkpoint_dir, encoder_dim
    )
    return LoadedAutoencoder(
        autoencoder=autoencoder,
        latent_mean=latent_mean,
        latent_std=latent_std,
        latent_covariance=latent_covariance,
        metadata=metadata,
        checkpoint_dir=checkpoint_dir,
    )


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
    autoencoder = nnx.merge(graphdef, state)

    train_latents = encode_dataset(
        autoencoder, train_data, batch_size=eval_batch_size
    )
    latent_mean, latent_std, latent_covariance = _compute_latent_statistics(
        train_latents
    )
    _write_latent_statistics(
        checkpoint_dir, latent_mean, latent_std, latent_covariance
    )

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
    return autoencoder, metadata


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
    latent_std = jnp.maximum(jnp.std(train_latents, axis=0), 1e-20)
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
    statistics_path = _latent_statistics_path(checkpoint_dir)
    if statistics_path.exists():
        latent_mean, latent_std, latent_covariance = _load_latent_statistics(
            checkpoint_dir, encoder_dim
        )
    else:
        latent_mean, latent_std, latent_covariance = _compute_latent_statistics(
            train_latents
        )
        _write_latent_statistics(
            checkpoint_dir, latent_mean, latent_std, latent_covariance
        )
    train_norm = (train_latents - latent_mean[None, :]) / latent_std[None, :]
    test_norm = (test_latents - latent_mean[None, :]) / latent_std[None, :]
    latent_log_det = float(jnp.log(latent_std).sum())
    return {
        "train": np.asarray(train_norm, dtype=np.float32),
        "test": np.asarray(test_norm, dtype=np.float32),
        "latent_mean": np.asarray(latent_mean, dtype=np.float32),
        "latent_std": np.asarray(latent_std, dtype=np.float32),
        "latent_covariance": np.asarray(latent_covariance, dtype=np.float32),
        "image_shape": image_shape,
        "eval_mode": "latent",
        "latent_log_det_per_example": latent_log_det,
        "encoder_dim": encoder_dim,
        "autoencoder": autoencoder,
        "autoencoder_checkpoint_dir": checkpoint_dir,
        "autoencoder_metadata": metadata,
    }


def _autoencoder_and_statistics(
    context: LoadedAutoencoder | dict[str, Any],
) -> tuple[AutoEncoder, jax.Array, jax.Array]:
    if isinstance(context, LoadedAutoencoder):
        return context.autoencoder, context.latent_mean, context.latent_std

    autoencoder = context.get("autoencoder")
    if autoencoder is None:
        autoencoder, _ = load_autoencoder_checkpoint(
            context["autoencoder_checkpoint_dir"],
            tuple(context["image_shape"]),
            int(context["encoder_dim"]),
        )
    return (
        autoencoder,
        jnp.asarray(context["latent_mean"]),
        jnp.asarray(context["latent_std"]),
    )


def encode_images_to_latents(
    images: np.ndarray | jax.Array,
    context: LoadedAutoencoder | dict[str, Any],
    *,
    normalized: bool = True,
    batch_size: int = 2048,
) -> jax.Array:
    """Encode images, returning training-normalized latents by default."""

    autoencoder, latent_mean, latent_std = _autoencoder_and_statistics(context)
    latents = encode_dataset(autoencoder, images, batch_size=batch_size)
    if normalized:
        latents = (latents - latent_mean[None, :]) / latent_std[None, :]
    return latents


def decode_latents_to_images(
    latents: np.ndarray | jax.Array,
    context: LoadedAutoencoder | dict[str, Any],
    *,
    normalized: bool = True,
) -> jax.Array:
    """Decode latents, reversing training normalization by default."""

    autoencoder, latent_mean, latent_std = _autoencoder_and_statistics(context)
    latents = jnp.asarray(latents, dtype=jnp.float32)
    if normalized:
        latents = latents * latent_std[None, :] + latent_mean[None, :]
    return jnp.clip(autoencoder.decode(latents), 0.0, 1.0)
