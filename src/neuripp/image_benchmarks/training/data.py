"""Restartable image and cached-latent batch streams."""

from __future__ import annotations

import math
from typing import Any

import jax
import numpy as np


class RestartableImageStream:
    def __init__(self, iterator):
        self.iterator = iterator
        self.epoch = 0
        self.batch_index = 0
        self._generator = None

    def _reset(self):
        self._generator = self.iterator.iter_epoch(
            self.epoch, start_batch=self.batch_index
        )

    def next_batch(self):
        if self._generator is None:
            self._reset()
        try:
            batch = next(self._generator)
        except StopIteration:
            self.epoch += 1
            self.batch_index = 0
            self._reset()
            batch = next(self._generator)
        self.batch_index += 1
        return batch["image"]

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_index": self.batch_index}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self.batch_index = int(state["batch_index"])
        self._generator = None


class RestartableLatentStream:
    def __init__(
        self,
        mean,
        std,
        *,
        batch_size: int,
        seed: int,
        sampling_seed: int | None = None,
        sample_posterior: bool = True,
        shuffle: bool,
        drop_last: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.seed = seed
        self.sampling_seed = seed if sampling_seed is None else sampling_seed
        self.sample_posterior = sample_posterior
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.batch_index = 0
        self._order = None

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return len(self.mean) // self.batch_size
        return math.ceil(len(self.mean) / self.batch_size)

    def _epoch_order(self):
        order = np.arange(len(self.mean))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        return order

    def next_batch(self):
        if self.batch_index >= self.batches_per_epoch:
            self.epoch += 1
            self.batch_index = 0
            self._order = None
        if self._order is None:
            self._order = self._epoch_order()
        start = self.batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self.mean))
        indices = self._order[start:stop]
        means = np.asarray(self.mean[indices])
        self.batch_index += 1
        if not self.sample_posterior:
            return means
        stds = np.asarray(self.std[indices])
        base_key = jax.random.fold_in(jax.random.key(self.sampling_seed), self.epoch)
        keys = jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
            jax.numpy.asarray(indices, dtype=jax.numpy.uint32)
        )
        noise = np.asarray(
            jax.vmap(lambda key: jax.random.normal(key, means.shape[1:]))(keys)
        )
        return means + stds * noise

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_index": self.batch_index}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self.batch_index = int(state["batch_index"])
        self._order = None
