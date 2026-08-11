"""Stable project split construction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def stable_order(identifiers: Sequence[str], seed: int) -> np.ndarray:
    """Return positions ordered by a process-independent SHA-256 key."""

    keyed = []
    for position, identifier in enumerate(identifiers):
        digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
        keyed.append((digest, position))
    keyed.sort()
    return np.asarray([position for _, position in keyed], dtype=np.int64)


def stable_index_order(length: int, seed: int) -> np.ndarray:
    """Order stable source indices without allocating Python strings/digests."""

    values = np.arange(length, dtype=np.uint64) + np.uint64(seed)
    # Vectorized SplitMix64 finalizer.  It is deterministic across processes and
    # independent of NumPy's random-generator implementation.
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return np.argsort(values, kind="stable")


def split_holdout(
    indices: Sequence[int] | np.ndarray,
    identifiers: Sequence[str] | None,
    validation_size: int | float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if identifiers is not None and len(indices) != len(identifiers):
        raise ValueError("indices and identifiers must have equal length")
    if isinstance(validation_size, float):
        if not 0.0 < validation_size < 1.0:
            raise ValueError("fractional validation_size must be in (0, 1)")
        count = max(1, int(round(len(indices) * validation_size)))
    else:
        count = int(validation_size)
    if count < 1 or count >= len(indices):
        raise ValueError(
            f"validation_size must be between 1 and {len(indices) - 1}; got {count}"
        )
    order = (
        stable_index_order(len(indices), seed)
        if identifiers is None
        else stable_order(identifiers, seed)
    )
    validation_positions = order[:count]
    train_positions = order[count:]
    return indices[train_positions], indices[validation_positions]


def split_fixed_counts(
    indices: Sequence[int] | np.ndarray,
    identifiers: Sequence[str] | None,
    counts: tuple[int, int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if sum(counts) != len(indices):
        raise ValueError(
            f"split counts {counts} sum to {sum(counts)}, expected {len(indices)}"
        )
    if identifiers is not None and len(indices) != len(identifiers):
        raise ValueError("indices and identifiers must have equal length")
    order = (
        stable_index_order(len(indices), seed)
        if identifiers is None
        else stable_order(identifiers, seed)
    )
    first, second, third = np.split(order, np.cumsum(counts)[:-1])
    return indices[first], indices[second], indices[third]


def write_indices(path: str | Path, indices: Iterable[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    array = (
        np.asarray(indices, dtype=np.int64)
        if isinstance(indices, np.ndarray)
        else np.asarray(list(indices), dtype=np.int64)
    )
    with temporary.open("wb") as stream:
        np.save(stream, array)
    temporary.replace(path)


def read_indices(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)
