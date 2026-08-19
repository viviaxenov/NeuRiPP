"""Config-driven unconditional Flow Matching image benchmark runner.

The parent process intentionally avoids importing JAX. Spawned workers set
``CUDA_VISIBLE_DEVICES`` before importing accelerator-dependent modules.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

# Keep benchmark-only modules local to examples rather than the installable package.
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from image_benchmarks.config import gpu_groups, load_config, plan_runs
from image_benchmarks.datasets.hf_loader import download_dataset
from image_benchmarks.datasets.manifest import DatasetManifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _dataset_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["problem"]["dataset"]
    return {
        "hf_token": dataset.get("hf_token"),
        "revision": dataset.get("revision"),
        "resolution": dataset["resolution"],
        "crop": dataset.get("crop"),
        "split_seed": int(dataset.get("split_seed", 20260811)),
        "train_size": dataset.get("train_size"),
        "offline": bool(dataset.get("offline", False)),
    }


def prepare_dataset(config: dict[str, Any]) -> DatasetManifest:
    dataset = config["problem"]["dataset"]
    return download_dataset(
        dataset["name"], dataset["cache_dir"], **_dataset_kwargs(config)
    )


def verify_manifest_batches(config: dict[str, Any], manifest: DatasetManifest) -> None:
    """Load one deterministic batch from every logical split after preparation."""

    from image_benchmarks.datasets.hf_loader import load_split

    for split in manifest.splits:
        iterator = load_split(
            manifest,
            split,
            batch_size=1,
            seed=config["experiment"]["seed"],
            shuffle=False,
            offline=config["problem"]["dataset"].get("offline", False),
        )
        batch = next(iter(iterator))
        if np.asarray(batch["image"]).shape != (
            1,
            *tuple(config["resolved"]["image_shape"]),
        ):
            raise ValueError(f"Prepared split {split!r} returned an unexpected image shape")


def prepare_assets(config: dict[str, Any]) -> dict[str, Any]:
    from image_benchmarks.assets.diffuse_nnx import require_diffuse_nnx
    from image_benchmarks.assets.files import (
        prepare_inception_weights,
        prepare_vae_checkpoint,
    )

    prepared: dict[str, Any] = {}
    encoder = config["problem"]["encoder"]
    uses_diffuse_nnx = (
        encoder["type"] == "vae"
        or config["rhs"]["type"] == "sit"
        or config["evaluation"]["fid"]["enabled"]
    )
    if uses_diffuse_nnx:
        prepared["diffuse_nnx_version"] = require_diffuse_nnx()

    if encoder["type"] == "vae":
        prepared["vae_checkpoint"] = prepare_vae_checkpoint(
            encoder["checkpoint"],
            auto_download=bool(encoder.get("auto_download", True)),
            expected_sha256=encoder.get("expected_sha256"),
        )
    if config["evaluation"]["fid"]["enabled"]:
        fid = config["evaluation"]["fid"]
        prepared["inception_weights"] = prepare_inception_weights(
            fid["weights_path"],
            auto_download=bool(fid.get("auto_download", True)),
            expected_sha256=fid["expected_sha256"],
        )
    return prepared


def _session_dir(config, run_name, output_dir, resume):
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    elif run_name:
        path = Path(config["experiment"]["output_root"]) / run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(config["experiment"]["output_root"]) / (
            f"{config['experiment']['name']}_{timestamp}"
        )
    if path.exists() and not resume:
        raise FileExistsError(f"Session already exists; use --resume: {path}")
    return path


def _validate_resume_session(session_dir, config, runs, manifest=None):
    resolved_path = session_dir / "resolved_config.json"
    planned_path = session_dir / "planned_runs.json"
    if not resolved_path.is_file() or not planned_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume session without resolved_config.json and planned_runs.json: {session_dir}"
        )
    saved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
    saved_runs = json.loads(planned_path.read_text(encoding="utf-8"))
    if saved_config != config:
        raise ValueError("Refusing to resume: resolved config differs from saved session")
    if saved_runs != runs:
        raise ValueError("Refusing to resume: planned runs differ from saved session")
    if manifest is not None:
        saved_manifest_path = session_dir / "dataset_manifest.json"
        if not saved_manifest_path.is_file():
            raise FileNotFoundError("Cannot resume session without dataset_manifest.json")
        saved_manifest = DatasetManifest.read(saved_manifest_path)
        if saved_manifest.digest != manifest.digest:
            raise ValueError("Refusing to resume: dataset manifest differs from saved session")


def _initialize_session(session_dir, config_path, config, runs, manifest, *, resume):
    if resume:
        _validate_resume_session(session_dir, config, runs, manifest)
        return
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "runs").mkdir(exist_ok=True)
    (session_dir / "plots").mkdir(exist_ok=True)
    (session_dir / "input_config.json").write_text(
        Path(config_path).read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_json(session_dir / "resolved_config.json", config)
    _write_json(session_dir / "planned_runs.json", runs)
    _write_json(session_dir / "dataset_manifest_summary.json", manifest.summary())
    _write_json(session_dir / "dataset_manifest.json", manifest.to_dict())


def _collect_images(iterator, count: int) -> tuple[np.ndarray, list[str]]:
    images = []
    identifiers = []
    for batch in iterator:
        remaining = count - sum(len(value) for value in images)
        if remaining <= 0:
            break
        take = min(remaining, len(batch["image"]))
        images.append(batch["image"][:take])
        identifiers.extend(batch["id"][:take])
    if not images:
        raise ValueError("Dataset split produced no images")
    return np.concatenate(images, axis=0), identifiers


def _build_encoder(config, manifest, seed):
    from image_benchmarks.datasets.hf_loader import load_split
    from image_benchmarks.encoders.registry import build_encoder

    encoder_config = config["problem"]["encoder"]
    image_shape = tuple(config["resolved"]["image_shape"])
    if encoder_config["type"] != "ae":
        return build_encoder(encoder_config, image_shape, seed=seed)
    checkpoint = Path(encoder_config["checkpoint"])
    lock = checkpoint.parent / f".{checkpoint.name}.lock"
    with _file_lock(lock):
        train_images = validation_images = None
        if not (checkpoint / "model").exists():
            train_iterator = load_split(
                manifest,
                "train",
                int(encoder_config.get("training", {}).get("batch_size", 1024)),
                seed,
                shuffle=False,
                offline=config["problem"]["dataset"].get("offline", False),
            )
            validation_name = "fm_validation" if "fm_validation" in manifest.splits else "validation"
            validation_iterator = load_split(
                manifest,
                validation_name,
                int(encoder_config.get("training", {}).get("eval_batch_size", 2048)),
                seed,
                shuffle=False,
                offline=config["problem"]["dataset"].get("offline", False),
            )
            train_images, _ = _collect_images(
                train_iterator, manifest.splits["train"].count
            )
            validation_images, _ = _collect_images(
                validation_iterator, manifest.splits[validation_name].count
            )
        return build_encoder(
            encoder_config,
            image_shape,
            seed=seed,
            train_images=train_images,
            validation_images=validation_images,
            dataset_name=manifest.name,
        )


def _ensure_latent_cache(config, manifest, encoder, split, seed):
    from image_benchmarks.datasets.hf_loader import load_split
    from image_benchmarks.encoders.cache import (
        LatentCacheKey,
        LatentCacheWriter,
        open_latent_cache,
    )

    encoder_config = config["problem"]["encoder"]
    key = LatentCacheKey(
        dataset_revision=manifest.hf_revision,
        split=split,
        resolution=manifest.resolution,
        crop=manifest.crop,
        encoder_checkpoint_sha256=encoder.checkpoint_sha256,
        latent_scale=float(getattr(encoder, "scale_factor", 1.0)),
        split_indices_sha256=manifest.splits[split].indices_sha256,
        dataset_manifest_digest=manifest.digest,
    )
    cache_root = Path(encoder_config["latent_cache_dir"])
    try:
        return open_latent_cache(cache_root, key)
    except FileNotFoundError:
        pass
    lock = cache_root / f".{key.digest}.lock"
    with _file_lock(lock):
        try:
            return open_latent_cache(cache_root, key)
        except FileNotFoundError:
            pass
        batch_size = int(encoder_config.get("encoding_batch_size", 32))
        iterator = load_split(
            manifest,
            split,
            batch_size,
            seed,
            shuffle=False,
            offline=config["problem"]["dataset"].get("offline", False),
        )
        writer = None
        try:
            for batch in iterator:
                images = batch["image"]
                if hasattr(encoder, "encode_stats"):
                    mean, std = encoder.encode_stats(images)
                else:
                    mean = encoder.encode(images)
                    std = np.zeros_like(np.asarray(mean))
                mean = np.asarray(mean, dtype=np.float32)
                std = np.asarray(std, dtype=np.float32)
                if writer is None:
                    writer = LatentCacheWriter(
                        cache_root,
                        key,
                        count=manifest.splits[split].count,
                        latent_shape=mean.shape[1:],
                        dtype=np.float32,
                    )
                writer.write_batch(mean, std, batch["id"])
            if writer is None:
                raise ValueError(f"Cannot cache empty split {split}")
            return writer.finalize()
        except Exception:
            if writer is not None:
                writer.abort()
            raise


def _make_stream(
    config,
    manifest,
    encoder,
    split,
    seed,
    *,
    train,
    sampling_seed=None,
    augmentation_seed=None,
):
    from image_benchmarks.datasets.hf_loader import load_split
    from image_benchmarks.training.data import (
        RestartableImageStream,
        RestartableLatentStream,
    )

    batch_size = config["training"]["batch_size"]
    encoder_config = config["problem"]["encoder"]
    if encoder_config["type"] == "none":
        iterator = load_split(
            manifest,
            split,
            batch_size,
            seed,
            shuffle=train,
            augmentation_seed=augmentation_seed,
            horizontal_flip=(
                train and bool(config["problem"]["dataset"].get("horizontal_flip", False))
            ),
            drop_last=train,
            offline=config["problem"]["dataset"].get("offline", False),
        )
        return RestartableImageStream(iterator)
    if not encoder_config.get("cache_latents", True):
        raise ValueError("Latent image benchmark runs currently require cache_latents=true")
    cache = _ensure_latent_cache(config, manifest, encoder, split, seed)
    mean, std, _ = cache.load(load_identifiers=False)
    return RestartableLatentStream(
        mean,
        std,
        batch_size=batch_size,
        seed=seed,
        sampling_seed=sampling_seed,
        sample_posterior=bool(encoder_config.get("sample_posterior", True)),
        shuffle=train,
        drop_last=train,
    )


def _fixed_validation(config, manifest, encoder, run):
    from image_benchmarks.datasets.hf_loader import load_split
    from image_benchmarks.evaluation.validation import make_fixed_fm_validation
    from image_benchmarks.training.data import RestartableLatentStream

    evaluation = config["evaluation"]["val_fm_loss"]
    count = min(evaluation["num_samples"], manifest.splits[
        "fm_validation" if "fm_validation" in manifest.splits else "validation"
    ].count)
    split = "fm_validation" if "fm_validation" in manifest.splits else "validation"
    encoder_config = config["problem"]["encoder"]
    if encoder_config["type"] == "none":
        iterator = load_split(
            manifest,
            split,
            evaluation["batch_size"],
            run["rng_seeds"]["evaluation"],
            shuffle=False,
            offline=config["problem"]["dataset"].get("offline", False),
        )
        states, identifiers = _collect_images(iterator, count)
    else:
        cache = _ensure_latent_cache(
            config, manifest, encoder, split, run["rng_seeds"]["encoder_sampling"]
        )
        mean, std, identifiers = cache.load(load_identifiers=True)
        stream = RestartableLatentStream(
            mean,
            std,
            batch_size=count,
            seed=run["rng_seeds"]["evaluation"],
            sampling_seed=run["rng_seeds"]["evaluation"],
            sample_posterior=bool(encoder_config.get("sample_posterior", True)),
            shuffle=False,
            drop_last=False,
        )
        states = stream.next_batch()[:count]
        identifiers = identifiers[:count]
    return make_fixed_fm_validation(
        states, identifiers, run["rng_seeds"]["evaluation"]
    )


def _real_sample_metric_states(config, manifest, encoder, run, count):
    """Collect deterministic evaluation examples in the model's state space."""

    from image_benchmarks.datasets.hf_loader import load_split
    from image_benchmarks.training.data import RestartableLatentStream

    split = config["evaluation"]["split"]
    metric_config = config["evaluation"]["sample_metrics"]
    encoder_config = config["problem"]["encoder"]
    if encoder_config["type"] == "none":
        iterator = load_split(
            manifest,
            split,
            metric_config["batch_size"],
            config["evaluation"]["seed"],
            shuffle=False,
            offline=config["problem"]["dataset"].get("offline", False),
        )
        states, _ = _collect_images(iterator, count)
        return states
    cache = _ensure_latent_cache(
        config, manifest, encoder, split, run["rng_seeds"]["encoder_sampling"]
    )
    mean, std, _ = cache.load(load_identifiers=False)
    stream = RestartableLatentStream(
        mean,
        std,
        batch_size=min(metric_config["batch_size"], count),
        seed=config["evaluation"]["seed"],
        sampling_seed=config["evaluation"]["seed"],
        sample_posterior=bool(encoder_config.get("sample_posterior", True)),
        shuffle=False,
        drop_last=False,
    )
    batches = []
    collected = 0
    while collected < count:
        batch = np.asarray(stream.next_batch())
        take = min(len(batch), count - collected)
        batches.append(batch[:take])
        collected += take
    return np.concatenate(batches)


def _checkpoint_dirs(run_dir):
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return []
    return sorted(
        [path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.startswith("step_")]
    )


def _save_checkpoint(run_dir, trainer, stream, keep):
    import orbax.checkpoint as ocp

    step = trainer.step_count
    root = run_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step:09d}"
    if destination.exists():
        return destination
    temporary = root / f".step_{step:09d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    payload = {
        "trainer": trainer.checkpoint_payload(),
        "data_stream": stream.state_dict(),
    }
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(temporary, payload)
    checkpointer.wait_until_finished()
    _write_json(
        temporary / "metadata.json",
        {"step": step, "written_at": _utc_now(), "format": "orbax-standard"},
    )
    temporary.rename(destination)
    for obsolete in _checkpoint_dirs(run_dir)[:-keep]:
        shutil.rmtree(obsolete)
    return destination


def _restore_latest(run_dir, trainer, stream):
    import orbax.checkpoint as ocp

    checkpoints = _checkpoint_dirs(run_dir)
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    target = {
        "trainer": trainer.checkpoint_payload(),
        "data_stream": stream.state_dict(),
    }
    restored = ocp.StandardCheckpointer().restore(latest, target=target)
    trainer.restore_checkpoint_payload(restored["trainer"])
    stream.load_state_dict(restored["data_stream"])
    return latest


def _save_ema_checkpoint(run_dir, trainer, keep):
    """Persist the exponential moving average weights in their own directory."""
    if not trainer.ema_enabled:
        return None
    import orbax.checkpoint as ocp

    step = trainer.step_count
    root = run_dir / "ema_checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step:09d}"
    if destination.exists():
        return destination
    temporary = root / f".step_{step:09d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    payload = {"ema": trainer.ema_checkpoint_payload()}
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(temporary, payload)
    checkpointer.wait_until_finished()
    _write_json(
        temporary / "metadata.json",
        {"step": step, "written_at": _utc_now(), "format": "orbax-standard"},
    )
    temporary.rename(destination)
    for obsolete in sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")]
    )[:-keep]:
        shutil.rmtree(obsolete)
    return destination


def _fid_kid_eval(
    model,
    *,
    run_identity,
    config,
    run,
    encoder,
    validation,
    real_cache,
    real_fid_key,
    extractor,
    fid_config,
    sampling,
    run_dir,
    step,
    epoch,
    wall_clock_train_s,
):
    """Run FM-loss + FID/KID checkpoint evaluation for one model variant."""
    from image_benchmarks.evaluation.evaluator import evaluate_checkpoint

    return evaluate_checkpoint(
        model=model,
        encoder=encoder,
        validation=validation,
        real_feature_cache=real_cache,
        real_fid_key=real_fid_key,
        fid_cache_root=fid_config["cache_dir"],
        fake_cache_root=run_dir / "fake_features",
        extractor=extractor,
        step=step,
        epoch=epoch,
        wall_clock_train_s=wall_clock_train_s,
        fm_batch_size=config["evaluation"]["val_fm_loss"]["batch_size"],
        num_fake=fid_config["num_samples_final"],
        sampling_batch_size=sampling["batch_size"],
        sampling_seed=run["rng_seeds"]["sampling"],
        sampling_config=sampling,
        kid_config=config["evaluation"]["kid"],
        run_identity=run_identity,
    )


def _sample_metric_eval(
    model,
    *,
    real_states,
    metric_count,
    config,
    run,
    encoder,
    sample_metric_config,
):
    """Compute MMD / sliced-Wasserstein for one model variant."""
    from image_benchmarks.evaluation.sample_metrics import evaluate_sample_metrics
    from image_benchmarks.evaluation.sampling import generate_state_batches

    generated_states = np.concatenate(
        list(
            generate_state_batches(
                model,
                num_samples=metric_count,
                batch_size=sample_metric_config["batch_size"],
                seed=run["rng_seeds"]["sampling"],
                ode_method=config["evaluation"]["sampling"]["method"],
                ode_steps=config["evaluation"]["sampling"]["steps"],
                ode_kwargs=config["evaluation"]["sampling"].get("kwargs", {}),
            )
        )
    )
    result = evaluate_sample_metrics(
        real_states,
        generated_states,
        sample_metric_config,
        seed=config["evaluation"]["seed"],
    )
    result["sample_metrics_state_space"] = config["problem"]["encoder"]["type"]
    result["sample_metrics_split"] = config["evaluation"]["split"]
    result["sample_metrics_seed"] = config["evaluation"]["seed"]
    result["sample_metrics_sampling_seed"] = run["rng_seeds"]["sampling"]
    return result


def _periodic_sw_eval(
    model,
    *,
    real_states,
    num_samples,
    batch_size,
    num_projections,
    config,
    run,
):
    """Compute periodic sliced-Wasserstein validation for one model variant.

    Returns a dict keyed by 'sliced_wasserstein' (and provenance) so it can be
    logged as a validation_sw record. Unlike the final sample_metrics block,
    MMD is not computed here.
    """
    from image_benchmarks.evaluation.sampling import generate_state_batches
    from ott.tools.sliced import sliced_wasserstein
    import jax
    import jax.numpy as jnp

    generated_states = np.concatenate(
        list(
            generate_state_batches(
                model,
                num_samples=num_samples,
                batch_size=batch_size,
                seed=run["rng_seeds"]["sampling"],
                ode_method=config["evaluation"]["sampling"]["method"],
                ode_steps=config["evaluation"]["sampling"]["steps"],
                ode_kwargs=config["evaluation"]["sampling"].get("kwargs", {}),
            )
        )
    )
    real = np.asarray(real_states).reshape(len(real_states), -1)
    generated = np.asarray(generated_states).reshape(len(generated_states), -1)
    if real.shape != generated.shape:
        raise ValueError(
            "Sliced-Wasserstein validation requires real and generated states "
            "of equal shape"
        )
    distance, _ = sliced_wasserstein(
        jnp.asarray(real, dtype=jnp.float32),
        jnp.asarray(generated, dtype=jnp.float32),
        n_proj=num_projections,
        rng=jax.random.key(config["evaluation"]["seed"]),
    )
    return {
        "sliced_wasserstein": float(distance),
        "sliced_wasserstein_num_projections": int(num_projections),
        "sliced_wasserstein_num_samples": int(num_samples),
        "sample_metrics_state_space": config["problem"]["encoder"]["type"],
        "sample_metrics_split": config["evaluation"]["split"],
        "sample_metrics_seed": config["evaluation"]["seed"],
        "sample_metrics_sampling_seed": run["rng_seeds"]["sampling"],
    }


def _environment_snapshot():
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python": sys.version,
        "packages": dict(sorted(packages.items())),
    }


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _save_sample_grid(run_dir, model, encoder, config, seed, *, filename="sample_grid.pdf"):
    from image_benchmarks.evaluation.sampling import generate_image_batches
    import matplotlib.pyplot as plt

    grid = config.get("plotting", {}).get("sample_grid", {})
    if not grid.get("enabled", True):
        return
    rows, columns = int(grid.get("rows", 4)), int(grid.get("columns", 4))
    sampling = config["evaluation"]["sampling"]
    images = next(
        generate_image_batches(
            model,
            encoder,
            num_samples=rows * columns,
            batch_size=rows * columns,
            seed=seed,
            ode_method=sampling["method"],
            ode_steps=sampling["steps"],
            ode_kwargs=sampling.get("kwargs", {}),
        )
    )
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for axis, image in zip(axes.flat, images, strict=True):
        axis.imshow(np.squeeze(image), cmap="gray" if image.shape[-1] == 1 else None)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(run_dir / filename, dpi=120, format="pdf")
    plt.close(figure)


def _save_run_diagnostics(run_dir: Path, run: dict[str, Any]) -> None:
    """Per-run diagnostic: training loss, validation loss, sliced-Wasserstein.

    Plotted vs optimizer iteration and vs training wall-clock, with raw (solid)
    and EMA (dashed) curves. Sliced-Wasserstein is only shown when periodic
    validation_sw records exist. Saved as PDF in the run's plots directory.
    """
    import matplotlib.pyplot as plt

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    training = [
        record
        for record in records
        if record.get("type") == "train"
        and record.get("loss") is not None
        and np.isfinite(record["loss"])
    ]
    validation = [
        record
        for record in records
        if record.get("type") == "validation" and record.get("val_fm_loss") is not None
    ]
    validation_ema = [
        record
        for record in records
        if record.get("type") == "validation_ema"
        and record.get("val_fm_loss") is not None
    ]
    sw = [
        record
        for record in records
        if record.get("type") == "validation_sw"
        and record.get("sliced_wasserstein") is not None
    ]
    sw_ema = [
        record
        for record in records
        if record.get("type") == "validation_sw_ema"
        and record.get("sliced_wasserstein") is not None
    ]
    if not (training or validation or sw):
        return

    x_keys = ("optimizer_step", "wall_clock_train_s")
    x_labels = ("Optimizer iteration", "Training wall-clock (s)")
    row_specs = []
    val_row = None
    sw_row = None
    if training:
        row_specs.append(("Training loss", "loss", training))
    if validation:
        val_row = len(row_specs)
        row_specs.append(("Validation loss", "val_fm_loss", validation))
    if sw:
        sw_row = len(row_specs)
        row_specs.append(("Sliced Wasserstein", "sliced_wasserstein", sw))

    n_rows = len(row_specs)
    figure, axes = plt.subplots(
        n_rows, 2, figsize=(12, 3.2 * n_rows), layout="constrained", squeeze=False
    )
    for row, (title, metric, series) in enumerate(row_specs):
        for col, (key, xlabel) in enumerate(zip(x_keys, x_labels, strict=True)):
            axis = axes[row][col]
            axis.plot(
                [record[key] for record in series],
                [record[metric] for record in series],
                label="raw",
                linewidth=1.2,
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel(title)
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            axis.legend()
    if validation_ema and val_row is not None:
        for col, _ in enumerate(x_keys):
            axes[val_row][col].plot(
                [record[x_keys[col]] for record in validation_ema],
                [record["val_fm_loss"] for record in validation_ema],
                label="EMA",
                linestyle="--",
                linewidth=1.2,
            )
    if sw_ema and sw_row is not None:
        for col, _ in enumerate(x_keys):
            axes[sw_row][col].plot(
                [record[x_keys[col]] for record in sw_ema],
                [record["sliced_wasserstein"] for record in sw_ema],
                label="EMA",
                linestyle="--",
                linewidth=1.2,
            )
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plots_dir / "diagnostics.pdf", format="pdf")
    plt.close(figure)


def _run_one(config, run, manifest_path, session_dir, gpu_group, resume):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from image_benchmarks.distributed import DataParallelContext
    from image_benchmarks.encoders.identity import IdentityEncoder
    from image_benchmarks.evaluation.evaluator import prepare_real_feature_cache
    from image_benchmarks.evaluation.fid import FIDCacheKey
    from image_benchmarks.evaluation.inception import DiffuseInceptionFeatures
    from image_benchmarks.evaluation.reconstruction import reconstruction_metrics
    from image_benchmarks.rhs.registry import build_rhs
    from image_benchmarks.training.trainer import ImageTrainer
    from neuripp.parametric_pushforward.flow_matching import FlowMatching, flow_matching_loss

    run_dir = Path(session_dir) / "runs" / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume and (run_dir / "status.json").is_file() and (
        run_dir / "final_summary.json"
    ).is_file():
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            return {"run_id": run["run_id"], "status": "completed", "resumed": True}
    _write_json(run_dir / "expanded_config.json", {"config": config, "run": run})
    _write_json(run_dir / "status.json", {"status": "running", "updated_at": _utc_now()})
    manifest = DatasetManifest.read(manifest_path)
    context = (
        DataParallelContext.create(
            expected_device_count=len(gpu_group) if gpu_group else 1
        )
        if config["resources"].get("data_parallel", True)
        else None
    )
    encoder = _build_encoder(config, manifest, run["rng_seeds"]["encoder_sampling"])
    train_stream = _make_stream(
        config,
        manifest,
        encoder,
        "train",
        run["rng_seeds"]["dataset_shuffle"],
        train=True,
        sampling_seed=run["rng_seeds"]["encoder_sampling"],
        augmentation_seed=run["rng_seeds"]["augmentation"],
    )
    initial_batch = train_stream.next_batch()
    model_rngs = nnx.Rngs(
        default=run["seed"],
        params=run["seed"],
        dropout=run["rng_seeds"]["model_dropout"],
    )
    rhs = build_rhs(
        config["rhs"], tuple(config["resolved"]["state_shape"]), rngs=model_rngs
    )
    sampling = config["evaluation"]["sampling"]
    model = FlowMatching(
        rhs,
        model_rngs,
        config["training"]["batch_size"],
        ode_method=sampling["method"],
        ode_nstep_max=sampling["steps"],
        ode_kwargs=sampling.get("kwargs", {}),
    )
    training_rngs = nnx.Rngs(
        default=run["rng_seeds"]["fm_noise"],
        fm_noise=run["rng_seeds"]["fm_noise"],
        fm_time=run["rng_seeds"]["fm_time"],
        model_dropout=run["rng_seeds"]["model_dropout"],
    )
    trainer = ImageTrainer(
        model,
        run["method"],
        flow_matching_loss,
        initial_batch,
        training_rngs,
        data_parallel=context,
        dataset_size=manifest.splits["train"].count,
        ema_config=config.get("ema"),
    )
    if trainer.method.initialization_updates == 0:
        # The first batch has not been consumed by Optax/NGD initialization.
        train_stream.load_state_dict({"epoch": 0, "batch_index": 0})
    restored_checkpoint = _restore_latest(run_dir, trainer, train_stream) if resume else None

    validation = _fixed_validation(config, manifest, encoder, run)
    sw_validation_config = config["evaluation"]["sw_validation"]
    sw_real_states = None
    if sw_validation_config["enabled"]:
        sw_real_states = _real_sample_metric_states(
            config,
            manifest,
            encoder,
            run,
            sw_validation_config["num_samples"],
        )
    metrics_path = run_dir / "metrics.jsonl"
    previous_records = (
        [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        if metrics_path.is_file()
        else []
    )
    if resume:
        retained_records = [
            record
            for record in previous_records
            if restored_checkpoint is not None
            and int(record.get("optimizer_step", record.get("step", -1)))
            <= trainer.step_count
        ]
        if retained_records != previous_records:
            temporary = metrics_path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(
                    json.dumps(record, sort_keys=True, default=_json_default) + "\n"
                    for record in retained_records
                ),
                encoding="utf-8",
            )
            temporary.replace(metrics_path)
            previous_records = retained_records
    previous_validation = [
        float(record["val_fm_loss"])
        for record in previous_records
        if record.get("type") in {"validation", "evaluation"}
        and record.get("val_fm_loss") is not None
    ]
    ema_previous_validation = [
        float(record["val_fm_loss"])
        for record in previous_records
        if record.get("type") == "validation_ema"
        and record.get("val_fm_loss") is not None
    ]
    best_validation = min(previous_validation, default=float("inf"))
    ema_best_validation = min(ema_previous_validation, default=float("inf"))
    final_validation = next(
        (
            float(record["val_fm_loss"])
            for record in reversed(previous_records)
            if record.get("type") == "validation"
            and record.get("val_fm_loss") is not None
        ),
        float("inf"),
    )
    ema_final_validation = next(
        (
            float(record["val_fm_loss"])
            for record in reversed(previous_records)
            if record.get("type") == "validation_ema"
            and record.get("val_fm_loss") is not None
        ),
        float("inf"),
    )
    sw_previous = [
        float(record["sliced_wasserstein"])
        for record in previous_records
        if record.get("type") == "validation_sw"
        and record.get("sliced_wasserstein") is not None
    ]
    ema_sw_previous = [
        float(record["sliced_wasserstein"])
        for record in previous_records
        if record.get("type") == "validation_sw_ema"
        and record.get("sliced_wasserstein") is not None
    ]
    best_sw = min(sw_previous, default=float("inf"))
    ema_best_sw = min(ema_sw_previous, default=float("inf"))
    final_sw = next(
        (
            float(record["sliced_wasserstein"])
            for record in reversed(previous_records)
            if record.get("type") == "validation_sw"
            and record.get("sliced_wasserstein") is not None
        ),
        float("inf"),
    )
    ema_final_sw = next(
        (
            float(record["sliced_wasserstein"])
            for record in reversed(previous_records)
            if record.get("type") == "validation_sw_ema"
            and record.get("sliced_wasserstein") is not None
        ),
        float("inf"),
    )
    training = config["training"]
    while trainer.step_count < training["max_steps"]:
        batch = train_stream.next_batch()
        values = trainer.step(batch)
        step = trainer.step_count
        if step % training["log_every"] == 0 or step == training["max_steps"]:
            record = {
                "type": "train",
                **trainer.accounting(),
                "loss": float(values[0]),
                "grad_norm": float(jnp.sqrt(jnp.maximum(values[1], 0.0))),
                "natural_grad_norm": (
                    float(jnp.sqrt(jnp.maximum(values[2], 0.0)))
                    if len(values) > 2
                    else None
                ),
            }
            _append_jsonl(metrics_path, record)
        if step % training["validation_every"] == 0 or step == training["max_steps"]:
            from image_benchmarks.evaluation.validation import evaluate_fixed_fm_loss

            start = time.perf_counter()
            val_loss = evaluate_fixed_fm_loss(
                trainer.model,
                validation,
                batch_size=config["evaluation"]["val_fm_loss"]["batch_size"],
            )
            trainer.record_evaluation_time(time.perf_counter() - start)
            best_validation = min(best_validation, val_loss)
            final_validation = val_loss
            _append_jsonl(
                metrics_path,
                {"type": "validation", **trainer.accounting(), "val_fm_loss": val_loss},
            )
            if trainer.ema_enabled:
                ema_start = time.perf_counter()
                ema_val_loss = evaluate_fixed_fm_loss(
                    trainer.ema_model,
                    validation,
                    batch_size=config["evaluation"]["val_fm_loss"]["batch_size"],
                )
                trainer.record_evaluation_time(time.perf_counter() - ema_start)
                ema_best_validation = min(ema_best_validation, ema_val_loss)
                ema_final_validation = ema_val_loss
                _append_jsonl(
                    metrics_path,
                    {
                        "type": "validation_ema",
                        **trainer.accounting(),
                        "val_fm_loss": ema_val_loss,
                    },
                )
            if sw_validation_config["enabled"] and sw_real_states is not None:
                sw_start = time.perf_counter()
                sw_result = _periodic_sw_eval(
                    trainer.model,
                    real_states=sw_real_states,
                    num_samples=sw_validation_config["num_samples"],
                    batch_size=sw_validation_config["batch_size"],
                    num_projections=sw_validation_config["num_projections"],
                    config=config,
                    run=run,
                )
                sw_value = sw_result["sliced_wasserstein"]
                trainer.record_evaluation_time(time.perf_counter() - sw_start)
                best_sw = min(best_sw, sw_value)
                final_sw = sw_value
                _append_jsonl(
                    metrics_path,
                    {"type": "validation_sw", **trainer.accounting(), **sw_result},
                )
                if trainer.ema_enabled:
                    ema_sw_start = time.perf_counter()
                    ema_sw_result = _periodic_sw_eval(
                        trainer.ema_model,
                        real_states=sw_real_states,
                        num_samples=sw_validation_config["num_samples"],
                        batch_size=sw_validation_config["batch_size"],
                        num_projections=sw_validation_config["num_projections"],
                        config=config,
                        run=run,
                    )
                    ema_sw_value = ema_sw_result["sliced_wasserstein"]
                    trainer.record_evaluation_time(
                        time.perf_counter() - ema_sw_start
                    )
                    ema_best_sw = min(ema_best_sw, ema_sw_value)
                    ema_final_sw = ema_sw_value
                    _append_jsonl(
                        metrics_path,
                        {
                            "type": "validation_sw_ema",
                            **trainer.accounting(),
                            **ema_sw_result,
                        },
                    )
        if step % training["checkpoint_every"] == 0:
            _save_checkpoint(
                run_dir, trainer, train_stream, training["keep_checkpoints"]
            )
            _save_ema_checkpoint(run_dir, trainer, training["keep_checkpoints"])
    checkpoint = _save_checkpoint(
        run_dir, trainer, train_stream, training["keep_checkpoints"]
    )
    ema_checkpoint = _save_ema_checkpoint(run_dir, trainer, training["keep_checkpoints"])
    if not np.isfinite(final_validation):
        from image_benchmarks.evaluation.validation import evaluate_fixed_fm_loss

        final_validation = evaluate_fixed_fm_loss(
            trainer.model,
            validation,
            batch_size=config["evaluation"]["val_fm_loss"]["batch_size"],
        )
        best_validation = min(best_validation, final_validation)
    if trainer.ema_enabled and not np.isfinite(ema_final_validation):
        from image_benchmarks.evaluation.validation import evaluate_fixed_fm_loss

        ema_final_validation = evaluate_fixed_fm_loss(
            trainer.ema_model,
            validation,
            batch_size=config["evaluation"]["val_fm_loss"]["batch_size"],
        )
        ema_best_validation = min(ema_best_validation, ema_final_validation)

    final_summary = {
        **trainer.accounting(),
        "val_fm_loss": final_validation,
        "final_val_fm_loss": final_validation,
        "best_val_fm_loss": best_validation,
        "checkpoint": str(checkpoint),
        "ema_val_fm_loss": ema_final_validation,
        "ema_final_val_fm_loss": ema_final_validation,
        "ema_best_val_fm_loss": ema_best_validation,
        "ema_checkpoint": str(ema_checkpoint) if ema_checkpoint is not None else None,
    }
    if sw_validation_config["enabled"]:
        final_summary["final_sw"] = (
            final_sw if np.isfinite(final_sw) else None
        )
        final_summary["best_sw"] = (
            best_sw if np.isfinite(best_sw) else None
        )
        final_summary["ema_final_sw"] = (
            ema_final_sw if np.isfinite(ema_final_sw) else None
        )
        final_summary["ema_best_sw"] = (
            ema_best_sw if np.isfinite(ema_best_sw) else None
        )
    fid_config = config["evaluation"]["fid"]
    if fid_config["enabled"]:
        extractor = DiffuseInceptionFeatures(
            fid_config["weights_path"],
            expected_sha256=fid_config["expected_sha256"],
        )
        evaluation_split = config["evaluation"]["split"]
        from image_benchmarks.datasets.hf_loader import load_split

        real_iterator = load_split(
            manifest,
            evaluation_split,
            int(fid_config.get("batch_size", 64)),
            config["evaluation"]["seed"],
            shuffle=False,
            offline=config["problem"]["dataset"].get("offline", False),
        )
        fid_key = FIDCacheKey(
            manifest.hf_revision,
            evaluation_split,
            manifest.resolution,
            manifest.crop,
            "gray_to_rgb" if manifest.channels == 1 else "rgb",
            feature_extractor=extractor.provenance,
            split_indices_sha256=manifest.splits[evaluation_split].indices_sha256,
            dataset_manifest_digest=manifest.digest,
        )
        real_cache = prepare_real_feature_cache(
            real_iterator,
            count=manifest.splits[evaluation_split].count,
            extractor=extractor,
            cache_root=fid_config["cache_dir"],
            key=fid_key,
        )
        result = _fid_kid_eval(
            trainer.model,
            run_identity=run["run_id"],
            config=config,
            run=run,
            encoder=encoder,
            validation=validation,
            real_cache=real_cache,
            real_fid_key=fid_key,
            extractor=extractor,
            fid_config=fid_config,
            sampling=sampling,
            run_dir=run_dir,
            step=trainer.step_count,
            epoch=trainer.effective_epoch or 0.0,
            wall_clock_train_s=trainer.wall_clock_train_s,
        )
        evaluation_duration = float(result["wall_clock_evaluation_s"])
        trainer.record_evaluation_time(evaluation_duration)
        result["evaluation_duration_s"] = evaluation_duration
        result["wall_clock_evaluation_s"] = trainer.wall_clock_evaluation_s
        final_summary.update(result)
        final_summary["final_val_fm_loss"] = result["val_fm_loss"]
        final_summary["best_val_fm_loss"] = min(
            best_validation, result["val_fm_loss"]
        )
        _append_jsonl(metrics_path, {"type": "evaluation", **result})
        if trainer.ema_enabled:
            ema_result = _fid_kid_eval(
                trainer.ema_model,
                run_identity=f"{run['run_id']}:ema",
                config=config,
                run=run,
                encoder=encoder,
                validation=validation,
                real_cache=real_cache,
                real_fid_key=fid_key,
                extractor=extractor,
                fid_config=fid_config,
                sampling=sampling,
                run_dir=run_dir,
                step=trainer.step_count,
                epoch=trainer.effective_epoch or 0.0,
                wall_clock_train_s=trainer.wall_clock_train_s,
            )
            ema_duration = float(ema_result["wall_clock_evaluation_s"])
            trainer.record_evaluation_time(ema_duration)
            ema_result["evaluation_duration_s"] = ema_duration
            ema_result["wall_clock_evaluation_s"] = trainer.wall_clock_evaluation_s
            ema_prefixed = {
                "step": ema_result["step"],
                "epoch": ema_result["epoch"],
                "wall_clock_train_s": ema_result["wall_clock_train_s"],
                "val_fm_loss": ema_result["val_fm_loss"],
            }
            for key in (
                "fid",
                "fid_num_fake",
                "fid_num_real",
                "kid_mean",
                "kid_std",
                "kid_stderr",
                "kid_subsets",
                "kid_subset_size",
                "evaluation_duration_s",
                "wall_clock_evaluation_s",
            ):
                if key in ema_result:
                    ema_prefixed[f"ema_{key}"] = ema_result[key]
            final_summary.update(ema_prefixed)
            final_summary["ema_final_val_fm_loss"] = ema_result["val_fm_loss"]
            final_summary["ema_best_val_fm_loss"] = min(
                ema_best_validation, ema_result["val_fm_loss"]
            )
            _append_jsonl(metrics_path, {"type": "evaluation_ema", **ema_result})

    sample_metric_config = config["evaluation"]["sample_metrics"]
    if any(
        sample_metric_config[name]["enabled"]
        for name in ("mmd", "sliced_wasserstein")
    ):
        start = time.perf_counter()
        metric_count = min(
            sample_metric_config["num_samples"],
            manifest.splits[config["evaluation"]["split"]].count,
        )
        if metric_count < 2:
            raise ValueError("Sample metric evaluation split must contain at least two examples")
        real_states = _real_sample_metric_states(
            config, manifest, encoder, run, metric_count
        )
        sample_metric_result = _sample_metric_eval(
            trainer.model,
            real_states=real_states,
            metric_count=metric_count,
            config=config,
            run=run,
            encoder=encoder,
            sample_metric_config=sample_metric_config,
        )
        duration = time.perf_counter() - start
        trainer.record_evaluation_time(duration)
        sample_metric_result["sample_metrics_duration_s"] = duration
        final_summary.update(sample_metric_result)
        _write_json(run_dir / "sample_metrics.json", sample_metric_result)
        _append_jsonl(
            metrics_path, {"type": "sample_metrics", **sample_metric_result}
        )
        if trainer.ema_enabled:
            ema_sample_start = time.perf_counter()
            ema_sample_metric_result = _sample_metric_eval(
                trainer.ema_model,
                real_states=real_states,
                metric_count=metric_count,
                config=config,
                run=run,
                encoder=encoder,
                sample_metric_config=sample_metric_config,
            )
            ema_duration = time.perf_counter() - ema_sample_start
            trainer.record_evaluation_time(ema_duration)
            ema_sample_metric_result["sample_metrics_duration_s"] = ema_duration
            ema_sample_prefixed = {
                f"ema_{key}": value
                for key, value in ema_sample_metric_result.items()
                if not key.startswith("sample_metrics_")
            }
            final_summary.update(ema_sample_prefixed)
            ema_sample_log = {
                "sample_metrics_ema": True,
                **ema_sample_metric_result,
            }
            _write_json(run_dir / "sample_metrics_ema.json", ema_sample_log)
            _append_jsonl(
                metrics_path, {"type": "sample_metrics_ema", **ema_sample_log}
            )
        if not sw_validation_config["enabled"]:
            # Stage-A style: only a final sliced-Wasserstein exists. Surface it in
            # the summary under the same keys used for periodic validation.
            if sample_metric_result.get("sliced_wasserstein") is not None:
                final_summary["final_sw"] = sample_metric_result[
                    "sliced_wasserstein"
                ]
                final_summary["best_sw"] = sample_metric_result[
                    "sliced_wasserstein"
                ]
            if (
                trainer.ema_enabled
                and ema_sample_metric_result.get("sliced_wasserstein") is not None
            ):
                final_summary["ema_final_sw"] = ema_sample_metric_result[
                    "sliced_wasserstein"
                ]
                final_summary["ema_best_sw"] = ema_sample_metric_result[
                    "sliced_wasserstein"
                ]

    if encoder.__class__.__name__ != "IdentityEncoder":
        from image_benchmarks.datasets.hf_loader import load_split

        recon_iterator = load_split(
            manifest,
            config["evaluation"]["split"],
            min(
                int(config["problem"]["encoder"].get("encoding_batch_size", 64)),
                manifest.splits[config["evaluation"]["split"]].count,
            ),
            config["evaluation"]["seed"],
            shuffle=False,
            offline=config["problem"]["dataset"].get("offline", False),
        )
        start = time.perf_counter()
        squared_error = 0.0
        element_count = 0
        for recon_batch in recon_iterator:
            images = np.asarray(recon_batch["image"])
            metrics = reconstruction_metrics(encoder, images)
            squared_error += metrics["encoder_recon_mse"] * images.size
            element_count += images.size
        if element_count == 0:
            raise ValueError("Reconstruction evaluation split is empty")
        recon_mse = squared_error / element_count
        final_summary.update(
            {
                "encoder_recon_mse": recon_mse,
                "encoder_recon_psnr": (
                    float("inf")
                    if recon_mse == 0.0
                    else float(10.0 * np.log10(4.0 / recon_mse))
                ),
                "encoder_recon_num_examples": manifest.splits[
                    config["evaluation"]["split"]
                ].count,
            }
        )
        trainer.record_evaluation_time(time.perf_counter() - start)
    start = time.perf_counter()
    _save_sample_grid(
        run_dir, trainer.model, encoder, config, run["rng_seeds"]["sampling"]
    )
    trainer.record_evaluation_time(time.perf_counter() - start)
    if trainer.ema_enabled:
        start = time.perf_counter()
        _save_sample_grid(
            run_dir,
            trainer.ema_model,
            encoder,
            config,
            run["rng_seeds"]["sampling"],
            filename="sample_grid_ema.pdf",
        )
        trainer.record_evaluation_time(time.perf_counter() - start)
    _save_run_diagnostics(run_dir, run)
    final_summary.update(trainer.accounting())
    _write_json(run_dir / "final_summary.json", final_summary)
    _write_json(
        run_dir / "metadata.json",
        {
            "git_commit": _git_commit(),
            "diffuse_nnx_commit": "da5f2b79497722931d279b012c90bec61050466b",
            "inception_weights_sha256": (
                fid_config.get("expected_sha256") if fid_config["enabled"] else None
            ),
            "dataset_manifest_digest": manifest.digest,
            "dataset_hf_id": manifest.hf_id,
            "dataset_hf_revision": manifest.hf_revision,
            "encoder_checkpoint_sha256": getattr(encoder, "checkpoint_sha256", None),
            "method": run["method"],
            "seeds": run["rng_seeds"],
            "ema": config.get("ema"),
            "devices": [str(device) for device in jax.devices()],
            "environment": _environment_snapshot(),
        },
    )
    _write_json(run_dir / "status.json", {"status": "completed", "updated_at": _utc_now()})
    return {"run_id": run["run_id"], "status": "completed"}


def _worker_loop(config, manifest_path, session_dir, gpu_group, task_queue, result_queue, resume):
    for key, value in config["resources"]["worker_env"].items():
        os.environ[key] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_group)
    while True:
        run = task_queue.get()
        if run is None:
            break
        try:
            result = _run_one(config, run, manifest_path, session_dir, gpu_group, resume)
        except Exception as error:
            run_dir = Path(session_dir) / "runs" / run["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            trace = traceback.format_exc()
            (run_dir / "error.txt").write_text(trace, encoding="utf-8")
            _write_json(
                run_dir / "status.json",
                {"status": "failed", "updated_at": _utc_now(), "error": str(error)},
            )
            result = {"run_id": run["run_id"], "status": "failed", "error": str(error)}
        result_queue.put(result)


def execute_runs(config, runs, manifest, session_dir, resume):
    groups = gpu_groups(config["resources"])
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    workers = []
    task_queue = context.Queue()
    for group in groups:
        process = context.Process(
            target=_worker_loop,
            args=(
                config,
                str(manifest.path),
                str(session_dir),
                group,
                task_queue,
                result_queue,
                resume,
            ),
        )
        process.start()
        workers.append(process)
    for run in runs:
        task_queue.put(run)
    for _ in workers:
        task_queue.put(None)

    results = []
    while len(results) < len(runs):
        try:
            results.append(result_queue.get(timeout=2.0))
        except queue.Empty:
            if not any(process.is_alive() for process in workers):
                break
    for process in workers:
        process.join()
    reported = {result["run_id"] for result in results}
    for run in runs:
        if run["run_id"] not in reported:
            run_dir = Path(session_dir) / "runs" / run["run_id"]
            status_path = run_dir / "status.json"
            if status_path.is_file() and (run_dir / "final_summary.json").is_file():
                durable_status = json.loads(status_path.read_text(encoding="utf-8"))
                if durable_status.get("status") == "completed":
                    results.append(
                        {
                            "run_id": run["run_id"],
                            "status": "completed",
                            "recovered_after_worker_exit": True,
                        }
                    )
                    continue
            result = {
                "run_id": run["run_id"],
                "status": "failed",
                "error": "worker exited before reporting the run",
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                run_dir / "status.json",
                {**result, "updated_at": _utc_now()},
            )
            (run_dir / "error.txt").write_text(result["error"] + "\n", encoding="utf-8")
            results.append(result)
    return results


def plot_session(session_dir: Path) -> None:
    import matplotlib.pyplot as plt

    planned = json.loads((session_dir / "planned_runs.json").read_text(encoding="utf-8"))

    # Per-method series keyed by metric: {"label": [(x_key, x_value), ...]}
    series = {
        "train_loss": [],  # (label, x_key, [(x, loss)])
        "val_loss": [],  # (label, ema_flag, x_key, [(x, val_loss)])
        "sw": [],  # (label, ema_flag, x_key, [(x, sw)])
    }
    x_keys = ("optimizer_step", "wall_clock_train_s")
    x_labels = ("Optimizer iteration", "Training wall-clock (s)")
    for run in planned:
        metrics_path = session_dir / "runs" / run["run_id"] / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        label = f"{run['method']['name']} r{run['restart_index']}"
        for key in x_keys:
            training = [
                record
                for record in records
                if record.get("type") == "train"
                and record.get("loss") is not None
                and np.isfinite(record["loss"])
            ]
            if training:
                series["train_loss"].append(
                    (
                        label,
                        False,
                        key,
                        [(record[key], record["loss"]) for record in training],
                    )
                )
            val = [
                record
                for record in records
                if record.get("type") == "validation"
                and record.get("val_fm_loss") is not None
            ]
            if val:
                series["val_loss"].append(
                    (
                        label,
                        False,
                        key,
                        [(record[key], record["val_fm_loss"]) for record in val],
                    )
                )
            val_ema = [
                record
                for record in records
                if record.get("type") == "validation_ema"
                and record.get("val_fm_loss") is not None
            ]
            if val_ema:
                series["val_loss"].append(
                    (
                        label,
                        True,
                        key,
                        [(record[key], record["val_fm_loss"]) for record in val_ema],
                    )
                )
            sw = [
                record
                for record in records
                if record.get("type") == "validation_sw"
                and record.get("sliced_wasserstein") is not None
            ]
            if sw:
                series["sw"].append(
                    (
                        label,
                        False,
                        key,
                        [
                            (record[key], record["sliced_wasserstein"])
                            for record in sw
                        ],
                    )
                )
            sw_ema = [
                record
                for record in records
                if record.get("type") == "validation_sw_ema"
                and record.get("sliced_wasserstein") is not None
            ]
            if sw_ema:
                series["sw"].append(
                    (
                        label,
                        True,
                        key,
                        [
                            (record[key], record["sliced_wasserstein"])
                            for record in sw_ema
                        ],
                    )
                )

    row_specs = [
        ("train_loss", "Training loss", True),
        ("val_loss", "Validation loss", True),
        ("sw", "Sliced Wasserstein", False),
    ]
    figure, axes = plt.subplots(
        len(row_specs), 2, figsize=(14, 3.6 * len(row_specs)), layout="constrained"
    )
    for row, (series_key, ylabel, log_scale) in enumerate(row_specs):
        for col, (key, xlabel) in enumerate(zip(x_keys, x_labels, strict=True)):
            axis = axes[row][col] if len(row_specs) > 1 else axes[col]
            for entry in series[series_key]:
                label, ema_flag, entry_key, points = entry
                if entry_key != key:
                    continue
                axis.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    label=(label + " EMA") if ema_flag else label,
                    linestyle="--" if ema_flag else "-",
                    linewidth=1.2,
                )
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if log_scale:
                axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend(fontsize="small")
    figure.savefig(session_dir / "plots" / "diagnostics_comparison.pdf", format="pdf")
    plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.plot_only and args.run_id is not None:
        raise ValueError("--plot-only and --run-id are mutually exclusive")
    config = load_config(args.config)
    runs = plan_runs(config)
    if args.run_id is not None:
        if args.run_id < 0 or args.run_id >= len(runs):
            raise ValueError(f"--run-id must be in 0..{len(runs) - 1}")
        runs = [runs[args.run_id]]
    print(json.dumps(config, indent=2, sort_keys=True))
    if args.validate_only:
        return 0
    session_dir = None
    if args.plot_only or not (args.prepare_assets or args.prepare_data):
        session_dir = _session_dir(
            config, args.run_name, args.output_dir, args.resume or args.plot_only
        )
    if args.plot_only:
        plot_session(session_dir)
        return 0
    if args.resume:
        _validate_resume_session(session_dir, config, runs)
    if args.prepare_assets:
        print(json.dumps(prepare_assets(config), indent=2, sort_keys=True))
        if not args.prepare_data:
            return 0
    elif not args.prepare_data:
        print(json.dumps(prepare_assets(config), indent=2, sort_keys=True))
    manifest = prepare_dataset(config)
    print(json.dumps(manifest.summary(), indent=2, sort_keys=True))
    if manifest.splits["train"].count < config["training"]["batch_size"]:
        raise ValueError(
            "training.batch_size exceeds the prepared training split size"
        )
    target_loader_epochs = config["training"].get("target_loader_epochs")
    if target_loader_epochs is not None:
        steps_per_loader_epoch = (
            manifest.splits["train"].count // config["training"]["batch_size"]
        )
        expected_steps = target_loader_epochs * steps_per_loader_epoch
        if config["training"]["max_steps"] != expected_steps:
            raise ValueError(
                "training.max_steps does not match target_loader_epochs with drop_last: "
                f"expected {expected_steps}, got {config['training']['max_steps']}"
            )
    if args.prepare_data:
        verify_manifest_batches(config, manifest)
        return 0
    _initialize_session(
        session_dir, args.config, config, runs, manifest, resume=args.resume
    )
    results = execute_runs(config, runs, manifest, session_dir, args.resume)
    summary = {
        "completed": sum(result["status"] == "completed" for result in results),
        "failed": sum(result["status"] != "completed" for result in results),
        "results": results,
    }
    _write_json(session_dir / "summary.json", summary)
    plot_session(session_dir)
    print(session_dir)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
