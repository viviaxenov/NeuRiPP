"""Disk-backed immutable Inception feature caches."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InceptionFeatureCache:
    def __init__(self, directory: Path, key: str, count: int, feature_dim: int):
        self.directory = directory
        self.key = key
        self.count = count
        self.feature_dim = feature_dim

    @property
    def features_path(self) -> Path:
        return self.directory / "features.npy"

    def load(self, *, verify_checksum: bool = False, mmap_mode: str = "r"):
        metadata = json.loads(
            (self.directory / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata["key"] != self.key:
            raise ValueError("Inception feature cache key mismatch")
        if verify_checksum and _sha256(self.features_path) != metadata["sha256"]:
            raise ValueError("Inception feature cache checksum mismatch")
        if verify_checksum and _sha256(self.directory / "ids.jsonl") != metadata["ids_sha256"]:
            raise ValueError("Inception feature identifier checksum mismatch")
        features = np.load(self.features_path, mmap_mode=mmap_mode, allow_pickle=False)
        if features.shape != (self.count, self.feature_dim):
            raise ValueError("Inception feature cache shape mismatch")
        return features


class InceptionFeatureWriter:
    def __init__(
        self,
        cache_root: str | Path,
        key: str,
        *,
        count: int,
        feature_dim: int = 2048,
    ):
        if count < 1 or feature_dim < 1:
            raise ValueError("Feature cache dimensions must be positive")
        self.root = Path(cache_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = str(key)
        self.digest = hashlib.sha256(self.key.encode("utf-8")).hexdigest()
        self.destination = self.root / self.digest
        self.temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.digest}.", dir=self.root)
        )
        self.count = int(count)
        self.feature_dim = int(feature_dim)
        self.features = np.lib.format.open_memmap(
            self.temporary / "features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(self.count, self.feature_dim),
        )
        self.identifiers = (self.temporary / "ids.jsonl").open("w", encoding="utf-8")
        self.position = 0
        self.closed = False

    def write_batch(self, features: Any, identifiers: Sequence[str]) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("Inception feature batch has the wrong shape")
        if len(features) != len(identifiers):
            raise ValueError("Feature batch and identifier counts differ")
        stop = self.position + len(features)
        if stop > self.count:
            raise ValueError("Feature writer received too many examples")
        self.features[self.position : stop] = features
        for identifier in identifiers:
            self.identifiers.write(json.dumps(str(identifier)) + "\n")
        self.position = stop

    def finalize(self) -> InceptionFeatureCache:
        if self.position != self.count:
            self.abort()
            raise ValueError(
                f"Feature writer expected {self.count} examples, got {self.position}"
            )
        self.features.flush()
        del self.features
        self.identifiers.close()
        checksum = _sha256(self.temporary / "features.npy")
        identifiers_checksum = _sha256(self.temporary / "ids.jsonl")
        metadata = {
            "key": self.key,
            "count": self.count,
            "feature_dim": self.feature_dim,
            "sha256": checksum,
            "ids_sha256": identifiers_checksum,
        }
        (self.temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            os.rename(self.temporary, self.destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            existing = json.loads(
                (self.destination / "metadata.json").read_text(encoding="utf-8")
            )
            if existing != metadata:
                shutil.rmtree(self.temporary)
                self.closed = True
                raise ValueError(f"Conflicting feature cache exists for {self.key}")
            shutil.rmtree(self.temporary)
        self.closed = True
        cache = open_feature_cache(self.root, self.key)
        cache.load(verify_checksum=True)
        return cache

    def abort(self) -> None:
        if self.closed:
            return
        if hasattr(self, "identifiers") and not self.identifiers.closed:
            self.identifiers.close()
        if hasattr(self, "features"):
            del self.features
        shutil.rmtree(self.temporary, ignore_errors=True)
        self.closed = True


def open_feature_cache(cache_root: str | Path, key: str) -> InceptionFeatureCache:
    root = Path(cache_root).expanduser().resolve()
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    directory = root / digest
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Inception feature cache not found for {key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return InceptionFeatureCache(
        directory,
        str(key),
        int(metadata["count"]),
        int(metadata["feature_dim"]),
    )
