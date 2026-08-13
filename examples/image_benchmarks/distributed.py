"""Replicated data-parallel placement for one benchmark run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from flax import nnx


@dataclass(frozen=True)
class DataParallelContext:
    mesh: Mesh
    replicated: NamedSharding
    data_sharding: NamedSharding
    device_count: int

    @classmethod
    def create(cls, expected_device_count: int | None = None):
        if jax.process_count() != 1:
            raise NotImplementedError(
                "Image benchmark data parallelism currently supports one JAX process "
                "per run; multi-host meshes are not implemented"
            )
        devices = np.asarray(jax.local_devices())
        if expected_device_count is not None and len(devices) != expected_device_count:
            raise RuntimeError(
                f"Worker expected {expected_device_count} JAX devices, got {len(devices)}: "
                f"{list(devices)}"
            )
        if len(devices) < 1:
            raise RuntimeError("No local JAX devices are available")
        mesh = Mesh(devices, ("data",))
        return cls(
            mesh=mesh,
            replicated=NamedSharding(mesh, P()),
            data_sharding=NamedSharding(mesh, P("data")),
            device_count=len(devices),
        )

    def replicate_graph_node(self, node: Any):
        graph, state = nnx.split(node)
        state = jax.tree.map(
            lambda value: jax.device_put(value, self.replicated), state
        )
        return nnx.merge(graph, state)

    def shard_batch(self, batch: Any):
        def place(value):
            value = np.asarray(value)
            if value.ndim < 1:
                raise ValueError("Every data-parallel batch leaf needs a batch axis")
            if value.shape[0] % self.device_count:
                raise ValueError(
                    f"Global batch size {value.shape[0]} is not divisible by "
                    f"{self.device_count} devices"
                )
            partition = P("data", *([None] * (value.ndim - 1)))
            return jax.device_put(value, NamedSharding(self.mesh, partition))

        return jax.tree.map(place, batch)

    def peak_memory_bytes(self) -> int | None:
        peaks = []
        for device in self.mesh.devices.flat:
            try:
                stats = device.memory_stats()
            except Exception:
                stats = None
            if not stats:
                continue
            value = stats.get("peak_bytes_in_use")
            if value is not None:
                peaks.append(int(value))
        return max(peaks) if peaks else None
