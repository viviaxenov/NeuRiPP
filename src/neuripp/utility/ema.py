"""Exponential moving average of an nnx module's parameters."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


@jax.jit
def ema_update_fn(shadow, current, decay):
    return jax.tree.map(
        lambda shadow_leaf, current_leaf: decay * shadow_leaf
        + (1.0 - decay) * current_leaf,
        shadow,
        current,
    )


class EMA:
    """Exponential moving average of an nnx module's parameters."""

    def __init__(self, model, *, decay: float, start_step: int = 0):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        if start_step < 0:
            raise ValueError(f"EMA start_step must be non-negative, got {start_step}")
        self.decay = float(decay)
        self.start_step = int(start_step)
        self._graph, shadow, self._rest = nnx.split(model, nnx.Param, ...)
        self.shadow = jax.tree.map(
            lambda value: jnp.array(value, copy=True), shadow
        )

    @property
    def model(self):
        return nnx.merge(self._graph, self.shadow, self._rest, copy=True)

    def update(self, model, step: int) -> None:
        if step < self.start_step:
            return
        current = nnx.state(model, nnx.Param)
        self.shadow = ema_update_fn(self.shadow, current, self.decay)

    def payload(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "shadow": jax.tree.map(
                lambda value: jnp.array(value, copy=True), self.shadow
            ),
        }

    def restore_payload(self, payload: dict[str, Any]) -> None:
        self.decay = float(payload["decay"])
        self.start_step = int(payload["start_step"])
        self.shadow = jax.tree.map(
            lambda value: jnp.array(value, copy=True), payload["shadow"]
        )
