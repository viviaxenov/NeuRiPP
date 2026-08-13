"""FID statistics, cache, and DiffuseNNX-compatible score calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import scipy.linalg

from image_benchmarks.assets.diffuse_nnx import prepare_diffuse_nnx_source


def _sqrtm_with_error_estimate(matrix: np.ndarray):
    """Normalize SciPy's pre/post-1.18 ``sqrtm`` return conventions."""

    try:
        result = scipy.linalg.sqrtm(matrix, disp=False)
    except TypeError:
        result = scipy.linalg.sqrtm(matrix)
    if isinstance(result, tuple):
        return result
    return result, None


@dataclass(frozen=True)
class FIDCacheKey:
    dataset_revision: str
    evaluation_split: str
    resolution: int
    crop: str
    channel_conversion: str
    dataset_resize: str = "pillow_lanczos"
    inception_resize: str = "jax_bilinear_299"
    feature_extractor: str = "diffuse_nnx_inception_v3_fid_023afd2"
    split_indices_sha256: str = ""
    dataset_manifest_digest: str = ""

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureStats:
    mu: np.ndarray
    sigma: np.ndarray
    count: int

    def as_diffuse_dict(self) -> dict[str, np.ndarray]:
        return {"mu": self.mu, "sigma": self.sigma}


class FeatureAccumulator:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sum_outer = None

    def update(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("Inception features must be a matrix")
        if self.sum is None:
            self.sum = np.zeros(features.shape[1], dtype=np.float64)
            self.sum_outer = np.zeros(
                (features.shape[1], features.shape[1]), dtype=np.float64
            )
        if features.shape[1] != len(self.sum):
            raise ValueError("Inception feature dimensions changed between batches")
        self.sum += features.sum(axis=0)
        self.sum_outer += features.T @ features
        self.count += len(features)

    def finalize(self) -> FeatureStats:
        if self.count < 2:
            raise ValueError("At least two feature vectors are required for FID")
        mu = self.sum / self.count
        sigma = (self.sum_outer - self.count * np.outer(mu, mu)) / (self.count - 1)
        return FeatureStats(mu, sigma, self.count)


def statistics_from_feature_batches(
    feature_batches: Iterable[np.ndarray],
) -> FeatureStats:
    accumulator = FeatureAccumulator()
    for features in feature_batches:
        accumulator.update(features)
    return accumulator.finalize()


def load_diffuse_fid_function(source_dir: str | Path):
    """Load the exact pinned ``calculate_fid`` function without Torch imports."""

    source_dir = prepare_diffuse_nnx_source(source_dir, auto_download=False)
    path = source_dir / "eval" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "calculate_fid"
        ),
        None,
    )
    if function is None:
        raise ImportError(f"Pinned DiffuseNNX calculate_fid not found in {path}")
    module = ast.Module(body=[function], type_ignores=[])
    scipy_compat = SimpleNamespace(
        linalg=SimpleNamespace(
            sqrtm=lambda matrix, disp=False: _sqrtm_with_error_estimate(matrix)
        )
    )
    namespace = {"np": np, "scipy": scipy_compat}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["calculate_fid"]


def calculate_fid(
    fake: FeatureStats,
    real: FeatureStats,
    *,
    diffuse_source_dir: str | Path | None = None,
) -> float:
    """Match ``diffuse_nnx.eval.utils.calculate_fid`` on cached statistics."""

    if fake.mu.shape != real.mu.shape or fake.sigma.shape != real.sigma.shape:
        raise ValueError("FID statistics have incompatible dimensions")
    if diffuse_source_dir is not None:
        reference = load_diffuse_fid_function(diffuse_source_dir)
        return reference(fake.as_diffuse_dict(), real.as_diffuse_dict())
    mean_term = np.square(fake.mu - real.mu).sum()
    covariance_root, _ = _sqrtm_with_error_estimate(fake.sigma @ real.sigma)
    score = mean_term + np.trace(fake.sigma + real.sigma - 2.0 * covariance_root)
    return float(np.real(score))


def write_fid_stats(
    cache_root: str | Path, key: FIDCacheKey, stats: FeatureStats
) -> Path:
    cache_root = Path(cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / f"{key.digest}.npz"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{key.digest}.", suffix=".npz", dir=cache_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(
            temporary,
            mu=stats.mu,
            sigma=stats.sigma,
            count=np.asarray(stats.count, dtype=np.int64),
            key_json=np.asarray(json.dumps(asdict(key), sort_keys=True)),
        )
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    loaded = load_fid_stats(cache_root, key)
    if loaded.count != stats.count or not np.array_equal(loaded.mu, stats.mu) or not np.array_equal(loaded.sigma, stats.sigma):
        raise ValueError(f"Conflicting FID statistics exist for key {key.digest}")
    return destination


def load_fid_stats(cache_root: str | Path, key: FIDCacheKey) -> FeatureStats:
    path = Path(cache_root).expanduser().resolve() / f"{key.digest}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No cached real FID statistics for key {key.digest}")
    with np.load(path, allow_pickle=False) as payload:
        stored_key = json.loads(str(payload["key_json"]))
        if stored_key != asdict(key):
            raise ValueError("FID cache key metadata mismatch")
        return FeatureStats(
            np.asarray(payload["mu"]),
            np.asarray(payload["sigma"]),
            int(payload["count"]),
        )
