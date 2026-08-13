"""Posterior-statistics caches for frozen stochastic encoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import errno
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LatentCacheKey:
    dataset_revision: str
    split: str
    resolution: int
    crop: str
    encoder_checkpoint_sha256: str
    latent_scale: float
    split_indices_sha256: str = ""
    dataset_manifest_digest: str = ""

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LatentStatsCache:
    directory: Path
    key: LatentCacheKey
    count: int

    @property
    def mean_path(self) -> Path:
        return self.directory / "mean.npy"

    @property
    def std_path(self) -> Path:
        return self.directory / "std.npy"

    @property
    def ids_path(self) -> Path:
        return self.directory / "ids.jsonl"

    def load(
        self,
        *,
        mmap_mode: str | None = "r",
        expected_identifiers: Sequence[str] | None = None,
        load_identifiers: bool = True,
        verify_checksums: bool = False,
    ):
        metadata = json.loads(
            (self.directory / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata["key_digest"] != self.key.digest:
            raise ValueError("Latent cache key mismatch")
        if verify_checksums:
            for filename, expected in metadata["sha256"].items():
                actual = _sha256_file(self.directory / filename)
                if actual != expected:
                    raise ValueError(
                        f"Latent cache checksum mismatch for {filename}: "
                        f"expected {expected}, got {actual}"
                    )
        mean = np.load(self.mean_path, mmap_mode=mmap_mode, allow_pickle=False)
        std = np.load(self.std_path, mmap_mode=mmap_mode, allow_pickle=False)
        identifiers = None
        if load_identifiers or expected_identifiers is not None:
            identifiers = [
                json.loads(line)
                for line in self.ids_path.read_text(encoding="utf-8").splitlines()
            ]
        expected_shape = tuple(metadata["shape"])
        if mean.shape != expected_shape or std.shape != expected_shape:
            raise ValueError("Latent cache shape mismatch")
        if str(mean.dtype) != metadata["dtype"] or str(std.dtype) != metadata["dtype"]:
            raise ValueError("Latent cache dtype mismatch")
        if len(mean) != self.count or (
            identifiers is not None and len(identifiers) != self.count
        ):
            raise ValueError("Latent cache count mismatch")
        if expected_identifiers is not None and list(expected_identifiers) != identifiers:
            raise ValueError("Latent cache identifier order mismatch")
        return mean, std, identifiers


class LatentCacheWriter:
    """Incrementally write large posterior caches using NumPy memmaps."""

    def __init__(
        self,
        cache_root: str | Path,
        key: LatentCacheKey,
        *,
        count: int,
        latent_shape: tuple[int, ...],
        dtype: str | np.dtype = np.float32,
    ):
        if count < 1:
            raise ValueError("Latent cache count must be positive")
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.count = int(count)
        self.shape = (self.count, *tuple(latent_shape))
        self.dtype = np.dtype(dtype)
        self.destination = self.cache_root / key.digest
        self.temporary = Path(
            tempfile.mkdtemp(prefix=f".{key.digest}.", dir=self.cache_root)
        )
        self.mean = np.lib.format.open_memmap(
            self.temporary / "mean.npy", mode="w+", dtype=self.dtype, shape=self.shape
        )
        self.std = np.lib.format.open_memmap(
            self.temporary / "std.npy", mode="w+", dtype=self.dtype, shape=self.shape
        )
        self._ids_stream = (self.temporary / "ids.jsonl").open("w", encoding="utf-8")
        self.position = 0
        self.closed = False

    def write_batch(
        self, mean: Any, std: Any, identifiers: Sequence[str]
    ) -> None:
        if self.closed:
            raise RuntimeError("Latent cache writer is closed")
        mean = np.asarray(mean, dtype=self.dtype)
        std = np.asarray(std, dtype=self.dtype)
        if mean.shape != std.shape or mean.shape[1:] != self.shape[1:]:
            raise ValueError("Latent cache batch shapes do not match the declared shape")
        if mean.shape[0] != len(identifiers):
            raise ValueError("Latent cache batch and identifiers have different counts")
        stop = self.position + mean.shape[0]
        if stop > self.count:
            raise ValueError("Latent cache writer received more examples than declared")
        self.mean[self.position : stop] = mean
        self.std[self.position : stop] = std
        for identifier in identifiers:
            self._ids_stream.write(json.dumps(str(identifier)) + "\n")
        self.position = stop

    def finalize(self) -> LatentStatsCache:
        if self.closed:
            raise RuntimeError("Latent cache writer is closed")
        if self.position != self.count:
            self.abort()
            raise ValueError(
                f"Latent cache expected {self.count} examples, received {self.position}"
            )
        self.mean.flush()
        self.std.flush()
        self._ids_stream.close()
        del self.mean, self.std
        checksums = {
            filename: _sha256_file(self.temporary / filename)
            for filename in ("mean.npy", "std.npy", "ids.jsonl")
        }
        metadata = {
            "key": asdict(self.key),
            "key_digest": self.key.digest,
            "count": self.count,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "sha256": checksums,
        }
        (self.temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            os.rename(self.temporary, self.destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            # Another worker published the same immutable cache first.
            existing_metadata = json.loads(
                (self.destination / "metadata.json").read_text(encoding="utf-8")
            )
            comparable = {
                "key_digest": self.key.digest,
                "count": self.count,
                "shape": list(self.shape),
                "dtype": str(self.dtype),
                "sha256": checksums,
            }
            if any(existing_metadata.get(key) != value for key, value in comparable.items()):
                shutil.rmtree(self.temporary)
                self.closed = True
                raise ValueError(
                    f"Conflicting latent cache already exists for key {self.key.digest}"
                )
            shutil.rmtree(self.temporary)
        self.closed = True
        cache = open_latent_cache(self.cache_root, self.key)
        cache.load(
            mmap_mode="r", load_identifiers=False, verify_checksums=True
        )
        return cache

    def abort(self) -> None:
        if self.closed:
            return
        if hasattr(self, "_ids_stream") and not self._ids_stream.closed:
            self._ids_stream.close()
        if hasattr(self, "mean"):
            del self.mean
        if hasattr(self, "std"):
            del self.std
        shutil.rmtree(self.temporary, ignore_errors=True)
        self.closed = True


def write_latent_cache(
    cache_root: str | Path,
    key: LatentCacheKey,
    mean: Any,
    std: Any,
    identifiers: Sequence[str],
) -> LatentStatsCache:
    """Small-array convenience wrapper around :class:`LatentCacheWriter`."""

    mean_array = np.asarray(mean)
    std_array = np.asarray(std)
    if mean_array.shape != std_array.shape or mean_array.ndim < 2:
        raise ValueError("Latent mean and std arrays must have matching batch shapes")
    writer = LatentCacheWriter(
        cache_root,
        key,
        count=mean_array.shape[0],
        latent_shape=mean_array.shape[1:],
        dtype=mean_array.dtype,
    )
    try:
        writer.write_batch(mean_array, std_array, identifiers)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


def open_latent_cache(
    cache_root: str | Path, key: LatentCacheKey
) -> LatentStatsCache:
    directory = Path(cache_root).expanduser().resolve() / key.digest
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Latent cache not found for key {key.digest}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["key_digest"] != key.digest:
        raise ValueError("Latent cache metadata key mismatch")
    return LatentStatsCache(directory, key, int(metadata["count"]))


def sample_cached_stats(mean: Any, std: Any, rng: jax.Array) -> jax.Array:
    mean = jnp.asarray(mean)
    std = jnp.asarray(std)
    return mean + std * jax.random.normal(rng, mean.shape)
