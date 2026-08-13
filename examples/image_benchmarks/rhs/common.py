"""Shared vector-field building blocks."""

from __future__ import annotations

import math

import jax.numpy as jnp


def sinusoidal_time_embedding(
    time: jnp.ndarray | float, dimension: int, max_period: float = 10000.0
) -> jnp.ndarray:
    if dimension < 2:
        raise ValueError("time embedding dimension must be at least two")
    half = dimension // 2
    frequencies = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / max(half, 1)
    )
    arguments = jnp.asarray(time, dtype=jnp.float32)[..., None] * frequencies
    embedding = jnp.concatenate((jnp.cos(arguments), jnp.sin(arguments)), axis=-1)
    if dimension % 2:
        embedding = jnp.pad(embedding, ((0, 1),))
    return embedding


def group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1
