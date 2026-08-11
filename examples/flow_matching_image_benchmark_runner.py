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

from neuripp.image_benchmarks.config import gpu_groups, load_config, plan_runs
from neuripp.image_benchmarks.datasets.hf_loader import download_dataset
from neuripp.image_benchmarks.datasets.manifest import DatasetManifest


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
        "validation_size": dataset.get("validation_size"),
        "offline": bool(dataset.get("offline", False)),
    }


def prepare_dataset(config: dict[str, Any]) -> DatasetManifest:
    dataset = config["problem"]["dataset"]
    return download_dataset(
        dataset["name"], dataset["cache_dir"], **_dataset_kwargs(config)
    )


def verify_manifest_batches(config: dict[str, Any], manifest: DatasetManifest) -> None:
    """Load one deterministic batch from every logical split after preparation."""

    from neuripp.image_benchmarks.datasets.hf_loader import load_split

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
    from neuripp.image_benchmarks.assets.diffuse_nnx import prepare_diffuse_nnx_source
    from neuripp.image_benchmarks.assets.files import prepare_vae_checkpoint

    prepared: dict[str, Any] = {}
    encoder = config["problem"]["encoder"]
    source_policies: dict[str, bool] = {}

    def require_source(path, auto_download):
        path = str(path)
        source_policies[path] = source_policies.get(path, True) and bool(auto_download)

    if encoder["type"] == "vae":
        require_source(
            encoder["source_dir"], encoder.get("source_auto_download", True)
        )
        prepared["vae_checkpoint"] = prepare_vae_checkpoint(
            encoder["checkpoint"],
            auto_download=bool(encoder.get("auto_download", True)),
            expected_sha256=encoder.get("expected_sha256"),
        )
    if config["rhs"]["type"] == "sit":
        require_source(
            config["rhs"]["source_dir"],
            config["rhs"].get("source_auto_download", True),
        )
    if config["evaluation"]["fid"]["enabled"]:
        require_source(
            config["evaluation"]["fid"]["source_dir"],
            config["evaluation"]["fid"].get("source_auto_download", True),
        )
    prepared["diffuse_nnx_sources"] = [
        str(
            prepare_diffuse_nnx_source(
                source, auto_download=source_policies[source]
            )
        )
        for source in sorted(source_policies)
    ]
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
    from neuripp.image_benchmarks.datasets.hf_loader import load_split
    from neuripp.image_benchmarks.encoders.registry import build_encoder

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
    from neuripp.image_benchmarks.datasets.hf_loader import load_split
    from neuripp.image_benchmarks.encoders.cache import (
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
    from neuripp.image_benchmarks.datasets.hf_loader import load_split
    from neuripp.image_benchmarks.training.data import (
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
    from neuripp.image_benchmarks.datasets.hf_loader import load_split
    from neuripp.image_benchmarks.evaluation.validation import make_fixed_fm_validation
    from neuripp.image_benchmarks.training.data import RestartableLatentStream

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


def _save_sample_grid(run_dir, model, encoder, config, seed):
    from neuripp.image_benchmarks.evaluation.sampling import generate_image_batches
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
    figure.savefig(run_dir / "sample_grid.png", dpi=120)
    plt.close(figure)


def _run_one(config, run, manifest_path, session_dir, gpu_group, resume):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from neuripp.image_benchmarks.distributed import DataParallelContext
    from neuripp.image_benchmarks.encoders.identity import IdentityEncoder
    from neuripp.image_benchmarks.evaluation.evaluator import (
        evaluate_checkpoint,
        prepare_real_feature_cache,
    )
    from neuripp.image_benchmarks.evaluation.fid import FIDCacheKey
    from neuripp.image_benchmarks.evaluation.inception import DiffuseInceptionFeatures
    from neuripp.image_benchmarks.evaluation.reconstruction import reconstruction_metrics
    from neuripp.image_benchmarks.rhs.registry import build_rhs
    from neuripp.image_benchmarks.training.trainer import ImageTrainer
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
    )
    if trainer.method.initialization_updates == 0:
        # The first batch has not been consumed by Optax/NGD initialization.
        train_stream.load_state_dict({"epoch": 0, "batch_index": 0})
    restored_checkpoint = _restore_latest(run_dir, trainer, train_stream) if resume else None

    validation = _fixed_validation(config, manifest, encoder, run)
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
        if record.get("val_fm_loss") is not None
    ]
    best_validation = min(previous_validation, default=float("inf"))
    final_validation = next(
        (
            float(record["val_fm_loss"])
            for record in reversed(previous_records)
            if record.get("type") == "validation"
            and record.get("val_fm_loss") is not None
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
            from neuripp.image_benchmarks.evaluation.validation import evaluate_fixed_fm_loss

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
        if step % training["checkpoint_every"] == 0:
            _save_checkpoint(
                run_dir, trainer, train_stream, training["keep_checkpoints"]
            )
    checkpoint = _save_checkpoint(
        run_dir, trainer, train_stream, training["keep_checkpoints"]
    )
    if not np.isfinite(final_validation):
        from neuripp.image_benchmarks.evaluation.validation import evaluate_fixed_fm_loss

        final_validation = evaluate_fixed_fm_loss(
            trainer.model,
            validation,
            batch_size=config["evaluation"]["val_fm_loss"]["batch_size"],
        )
        best_validation = min(best_validation, final_validation)

    final_summary = {
        **trainer.accounting(),
        "val_fm_loss": final_validation,
        "final_val_fm_loss": final_validation,
        "best_val_fm_loss": best_validation,
        "checkpoint": str(checkpoint),
    }
    fid_config = config["evaluation"]["fid"]
    if fid_config["enabled"]:
        extractor = DiffuseInceptionFeatures(fid_config["source_dir"])
        evaluation_split = config["evaluation"]["split"]
        from neuripp.image_benchmarks.datasets.hf_loader import load_split

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
        result = evaluate_checkpoint(
            model=trainer.model,
            encoder=encoder,
            validation=validation,
            real_feature_cache=real_cache,
            real_fid_key=fid_key,
            fid_cache_root=fid_config["cache_dir"],
            fake_cache_root=run_dir / "fake_features",
            extractor=extractor,
            diffuse_source_dir=fid_config["source_dir"],
            step=trainer.step_count,
            epoch=trainer.effective_epoch or 0.0,
            wall_clock_train_s=trainer.wall_clock_train_s,
            fm_batch_size=config["evaluation"]["val_fm_loss"]["batch_size"],
            num_fake=fid_config["num_samples_final"],
            sampling_batch_size=sampling["batch_size"],
            sampling_seed=run["rng_seeds"]["sampling"],
            sampling_config=sampling,
            kid_config=config["evaluation"]["kid"],
            run_identity=run["run_id"],
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

    if encoder.__class__.__name__ != "IdentityEncoder":
        from neuripp.image_benchmarks.datasets.hf_loader import load_split

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
    final_summary.update(trainer.accounting())
    _write_json(run_dir / "final_summary.json", final_summary)
    _write_json(
        run_dir / "metadata.json",
        {
            "git_commit": _git_commit(),
            "diffuse_nnx_commit": "023afd23c7b62a8cdb00e840b36a4ab8fc970bba",
            "dataset_manifest_digest": manifest.digest,
            "dataset_hf_id": manifest.hf_id,
            "dataset_hf_revision": manifest.hf_revision,
            "encoder_checkpoint_sha256": getattr(encoder, "checkpoint_sha256", None),
            "method": run["method"],
            "seeds": run["rng_seeds"],
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
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    plotted = False
    for run in planned:
        metrics_path = session_dir / "runs" / run["run_id"] / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        validation = [record for record in records if record.get("type") == "validation"]
        if validation:
            label = f"{run['method']['name']} r{run['restart_index']}"
            axes[0, 0].plot(
                [record["effective_epoch"] for record in validation],
                [record["val_fm_loss"] for record in validation],
                label=label,
            )
            axes[0, 1].plot(
                [record["wall_clock_train_s"] for record in validation],
                [record["val_fm_loss"] for record in validation],
                label=label,
            )
            plotted = True
        evaluations = [record for record in records if record.get("type") == "evaluation"]
        if evaluations:
            label = f"{run['method']['name']} r{run['restart_index']}"
            for metric in ("fid", "kid_mean"):
                points = [record for record in evaluations if record.get(metric) is not None]
                if points:
                    marker = "o" if metric == "fid" else "s"
                    axes[1, 0].plot(
                        [record["epoch"] for record in points],
                        [record[metric] for record in points],
                        marker=marker,
                        label=f"{label} {metric}",
                    )
                    axes[1, 1].plot(
                        [record["wall_clock_train_s"] for record in points],
                        [record[metric] for record in points],
                        marker=marker,
                        label=f"{label} {metric}",
                    )
    for axis, xlabel in zip(
        axes[0], ("Effective epoch", "Training wall-clock (s)"), strict=True
    ):
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Validation FM loss")
        axis.set_yscale("log")
        if plotted:
            axis.legend()
    for axis, xlabel in zip(
        axes[1], ("Effective epoch", "Training wall-clock (s)"), strict=True
    ):
        axis.set_xlabel(xlabel)
        axis.set_ylabel("FID / KID")
        if axis.lines:
            axis.legend()
    figure.savefig(session_dir / "plots" / "validation_fm_loss.png", dpi=140)
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
