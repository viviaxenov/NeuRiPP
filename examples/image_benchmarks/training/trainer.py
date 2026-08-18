"""Thin training state wrapper shared by all image optimizers."""

from __future__ import annotations

from typing import Any
import time
import numbers

import jax
import jax.numpy as jnp
from flax import nnx

from image_benchmarks.distributed import DataParallelContext
from image_benchmarks.training.methods import ResolvedMethod, resolve_method
from neuripp.utility.ema import EMA


class ImageTrainer:
    @staticmethod
    def _dynamic_scalars(value):
        """Keep method-state graph structure stable before and after JIT."""

        if isinstance(value, dict):
            return {
                key: ImageTrainer._dynamic_scalars(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(ImageTrainer._dynamic_scalars(item) for item in value)
        if isinstance(value, list):
            return [ImageTrainer._dynamic_scalars(item) for item in value]
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            return jnp.asarray(value)
        return value

    def __init__(
        self,
        model,
        method_config: dict[str, Any],
        loss,
        initial_batch,
        rngs: nnx.Rngs,
        *,
        data_parallel: DataParallelContext | None = None,
        dataset_size: int | None = None,
        ema_config: dict[str, Any] | None = None,
    ):
        self.data_parallel = data_parallel
        if data_parallel is not None:
            model = data_parallel.replicate_graph_node(model)
            rngs = data_parallel.replicate_graph_node(rngs)
            initial_batch = data_parallel.shard_batch(initial_batch)
        self.rngs = rngs
        self.method: ResolvedMethod = resolve_method(method_config, loss)
        initialization_start = time.perf_counter()
        self.state = self.method.init_fn(
            model,
            self.method.optimizer_args,
            self.method.optimizer_kwargs,
            initial_batch,
            rngs,
        )
        self.state = self._dynamic_scalars(self.state)
        if data_parallel is not None:
            self.state = data_parallel.replicate_graph_node(self.state)
        self._ema: EMA | None = None
        if ema_config and ema_config.get("enabled", True):
            self._ema = EMA(
                self.model,
                decay=float(ema_config.get("decay", 0.9999)),
                start_step=int(ema_config.get("start_step", 0)),
            )
        self._step = nnx.jit(self.method.step_fn)
        initial_batch_size = self._batch_size(initial_batch)
        self.step_count = self.method.initialization_updates
        self.examples_seen = self.method.initialization_updates * initial_batch_size
        self.dataset_size = dataset_size
        self.wall_clock_train_s = time.perf_counter() - initialization_start
        self.wall_clock_evaluation_s = 0.0

    @property
    def model(self):
        return self.state[0]

    @property
    def ema_enabled(self) -> bool:
        return self._ema is not None

    @property
    def ema_model(self):
        if self._ema is None:
            return None
        return self._ema.model

    def ema_checkpoint_payload(self) -> dict[str, Any] | None:
        if self._ema is None:
            return None
        return {
            **self._ema.payload(),
            "step_count": self.step_count,
            "examples_seen": self.examples_seen,
        }

    def step(self, batch):
        batch_size = self._batch_size(batch)
        if self.data_parallel is not None:
            batch = self.data_parallel.shard_batch(batch)
        start = time.perf_counter()
        self.state, values = self._step(self.state, batch, self.rngs)
        jax.block_until_ready(values)
        if self._ema is not None:
            self._ema.update(self.model, self.step_count + 1)
        self.wall_clock_train_s += time.perf_counter() - start
        self.step_count += 1
        self.examples_seen += batch_size
        return values

    @staticmethod
    def _batch_size(batch) -> int:
        leaves = jax.tree.leaves(batch)
        if not leaves or not hasattr(leaves[0], "shape") or not leaves[0].shape:
            raise ValueError("Training batch has no leading batch dimension")
        count = int(leaves[0].shape[0])
        if any(int(leaf.shape[0]) != count for leaf in leaves):
            raise ValueError("Training batch leaves have inconsistent batch sizes")
        return count

    @property
    def effective_epoch(self) -> float | None:
        if not self.dataset_size:
            return None
        return self.examples_seen / self.dataset_size

    def record_evaluation_time(self, seconds: float) -> None:
        self.wall_clock_evaluation_s += float(seconds)

    def checkpoint_payload(self) -> dict[str, Any]:
        _, state = nnx.split((self.state, self.rngs))
        state = jax.tree.map(lambda value: jnp.array(value, copy=True), state)
        return {
            "nnx_state": state,
            "step_count": self.step_count,
            "examples_seen": self.examples_seen,
            "wall_clock_train_s": self.wall_clock_train_s,
            "wall_clock_evaluation_s": self.wall_clock_evaluation_s,
            "ema": self._ema.payload() if self._ema is not None else None,
        }

    def restore_checkpoint_payload(self, payload: dict[str, Any]) -> None:
        graph, _ = nnx.split((self.state, self.rngs))
        self.state, self.rngs = nnx.merge(
            graph, payload["nnx_state"], copy=True
        )
        self.step_count = int(payload["step_count"])
        self.examples_seen = int(payload["examples_seen"])
        self.wall_clock_train_s = float(payload["wall_clock_train_s"])
        self.wall_clock_evaluation_s = float(payload["wall_clock_evaluation_s"])
        ema_payload = payload.get("ema")
        if self._ema is not None and ema_payload is not None:
            self._ema.restore_payload(ema_payload)

    def accounting(self) -> dict[str, Any]:
        parameters = nnx.state(self.model, nnx.Param)
        parameter_count = sum(
            int(value.size) for value in jax.tree.leaves(parameters)
        )
        return {
            "optimizer_step": self.step_count,
            "effective_epoch": self.effective_epoch,
            "examples_seen": self.examples_seen,
            "wall_clock_train_s": self.wall_clock_train_s,
            "wall_clock_evaluation_s": self.wall_clock_evaluation_s,
            "parameter_count": parameter_count,
            "ema_enabled": self.ema_enabled,
            "ema_decay": self._ema.decay if self._ema is not None else None,
            "ema_start_step": (
                self._ema.start_step if self._ema is not None else None
            ),
            "peak_accelerator_memory_bytes": (
                self.data_parallel.peak_memory_bytes()
                if self.data_parallel is not None
                else None
            ),
            **self.compute_counters(),
        }

    def compute_counters(self) -> dict[str, int | None]:
        # Existing method APIs do not expose reliable operation counts.
        return {
            "forward_evaluations": None,
            "gradient_evaluations": None,
            "jvp_evaluations": None,
            "vjp_evaluations": None,
        }
