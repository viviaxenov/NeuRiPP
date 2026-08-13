"""End-to-end image checkpoint evaluation orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from image_benchmarks.datasets.transforms import model_to_evaluation
from image_benchmarks.evaluation.features import (
    InceptionFeatureCache,
    InceptionFeatureWriter,
    open_feature_cache,
)
from image_benchmarks.evaluation.fid import (
    FIDCacheKey,
    FeatureStats,
    calculate_fid,
    load_fid_stats,
    statistics_from_feature_batches,
    write_fid_stats,
)
from image_benchmarks.evaluation.kid import calculate_kid
from image_benchmarks.evaluation.sampling import generate_image_batches
from image_benchmarks.evaluation.validation import (
    FixedFMValidationSet,
    evaluate_fixed_fm_loss,
)


def _feature_batches(features: np.ndarray, batch_size: int = 4096):
    for start in range(0, len(features), batch_size):
        yield np.asarray(features[start : start + batch_size])


def prepare_real_feature_cache(
    image_batches: Iterable[dict[str, Any] | np.ndarray],
    *,
    count: int,
    extractor,
    cache_root: str | Path,
    key: FIDCacheKey,
) -> InceptionFeatureCache:
    """Extract and cache held-out real features once per preprocessing key."""

    if getattr(extractor, "provenance", None) != key.feature_extractor:
        raise ValueError(
            "Inception extractor provenance does not match the FID cache key"
        )
    feature_root = Path(cache_root) / "features"
    try:
        existing = open_feature_cache(feature_root, key.digest)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if existing.count != count:
            raise ValueError("Cached real feature count does not match requested count")
        existing.load(verify_checksum=True, mmap_mode="r")
        try:
            load_fid_stats(Path(cache_root) / "stats", key)
        except FileNotFoundError:
            features = existing.load(mmap_mode="r")
            stats = statistics_from_feature_batches(_feature_batches(features))
            write_fid_stats(Path(cache_root) / "stats", key, stats)
        return existing
    writer = InceptionFeatureWriter(
        feature_root, key.digest, count=count, feature_dim=2048
    )
    try:
        for batch_index, batch in enumerate(image_batches):
            if isinstance(batch, dict):
                images = np.asarray(batch["image"])
                identifiers = batch.get("id")
            else:
                images = np.asarray(batch)
                identifiers = None
            if images.dtype != np.uint8:
                images = model_to_evaluation(images)
            if identifiers is None:
                identifiers = [
                    f"real:{batch_index}:{index}" for index in range(len(images))
                ]
            writer.write_batch(extractor(images), identifiers)
        cache = writer.finalize()
    except Exception:
        writer.abort()
        raise
    features = cache.load(mmap_mode="r", verify_checksum=True)
    stats = statistics_from_feature_batches(_feature_batches(features))
    write_fid_stats(Path(cache_root) / "stats", key, stats)
    return cache


def _fake_cache_key(
    *,
    run_identity: str,
    encoder_identity: str | None,
    extractor_provenance: str,
    step: int,
    seed: int,
    num_samples: int,
    sampling_config: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "step": step,
            "run_identity": run_identity,
            "encoder_identity": encoder_identity,
            "extractor": extractor_provenance,
            "seed": seed,
            "num_samples": num_samples,
            "sampling": sampling_config,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "fake:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_checkpoint(
    *,
    model,
    encoder,
    validation: FixedFMValidationSet,
    real_feature_cache: InceptionFeatureCache,
    real_fid_key: FIDCacheKey,
    fid_cache_root: str | Path,
    fake_cache_root: str | Path,
    extractor,
    diffuse_source_dir: str | Path | None,
    step: int,
    epoch: float,
    wall_clock_train_s: float,
    fm_batch_size: int,
    num_fake: int,
    sampling_batch_size: int,
    sampling_seed: int,
    sampling_config: dict[str, Any],
    kid_config: dict[str, Any] | None = None,
    run_identity: str,
) -> dict[str, Any]:
    """Evaluate FM loss, decode samples, and calculate FID/KID."""

    evaluation_start = time.perf_counter()
    val_fm_loss = evaluate_fixed_fm_loss(
        model, validation, batch_size=fm_batch_size
    )
    if real_feature_cache.key != real_fid_key.digest:
        raise ValueError("Real FID statistics and feature cache keys do not match")
    if getattr(extractor, "provenance", None) != real_fid_key.feature_extractor:
        raise ValueError("Evaluation extractor does not match real FID provenance")
    fake_key = _fake_cache_key(
        run_identity=run_identity,
        encoder_identity=getattr(encoder, "checkpoint_sha256", None),
        extractor_provenance=extractor.provenance,
        step=step,
        seed=sampling_seed,
        num_samples=num_fake,
        sampling_config=sampling_config,
    )
    writer = InceptionFeatureWriter(
        fake_cache_root, fake_key, count=num_fake, feature_dim=2048
    )
    position = 0
    try:
        for images in generate_image_batches(
            model,
            encoder,
            num_samples=num_fake,
            batch_size=sampling_batch_size,
            seed=sampling_seed,
            ode_method=sampling_config.get("method"),
            ode_steps=sampling_config.get("steps"),
            ode_kwargs=sampling_config.get("kwargs"),
        ):
            identifiers = [f"fake:{index}" for index in range(position, position + len(images))]
            writer.write_batch(extractor(images), identifiers)
            position += len(images)
        fake_cache = writer.finalize()
    except Exception:
        writer.abort()
        raise

    real_features = real_feature_cache.load(mmap_mode="r", verify_checksum=True)
    fake_features = fake_cache.load(mmap_mode="r", verify_checksum=True)
    real_stats = load_fid_stats(Path(fid_cache_root) / "stats", real_fid_key)
    fake_stats = statistics_from_feature_batches(_feature_batches(fake_features))
    result: dict[str, Any] = {
        "step": int(step),
        "epoch": float(epoch),
        "wall_clock_train_s": float(wall_clock_train_s),
        "val_fm_loss": float(val_fm_loss),
        "fid": calculate_fid(
            fake_stats, real_stats, diffuse_source_dir=diffuse_source_dir
        ),
        "fid_num_fake": int(fake_stats.count),
        "fid_num_real": int(real_stats.count),
    }
    if kid_config and kid_config.get("enabled", True):
        result.update(
            calculate_kid(
                real_features,
                fake_features,
                subsets=int(kid_config.get("subsets", 100)),
                subset_size=int(kid_config.get("subset_size", 1000)),
                seed=int(kid_config.get("seed", sampling_seed)),
            )
        )
    result["wall_clock_evaluation_s"] = time.perf_counter() - evaluation_start
    return result
