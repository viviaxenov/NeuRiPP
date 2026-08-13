"""Deterministic model-state and decoded-image generation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
from flax import nnx

from neuripp._ode._ode import solve_ode_batched
from image_benchmarks.datasets.transforms import model_to_evaluation


def generate_image_batches(
    model,
    encoder,
    *,
    num_samples: int,
    batch_size: int,
    seed: int,
    ode_method: str | None = None,
    ode_steps: int | None = None,
    ode_kwargs: dict[str, Any] | None = None,
) -> Iterator[np.ndarray]:
    if num_samples < 1 or batch_size < 1:
        raise ValueError("Sampling counts must be positive")
    method = model.ode_method if ode_method is None else ode_method
    steps = model.ode_nstep_max if ode_steps is None else ode_steps
    solver_kwargs = model.ode_kwargs if ode_kwargs is None else ode_kwargs
    base_key = jax.random.key(seed)
    for start in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples - start)
        sample_indices = jax.numpy.arange(start, start + count, dtype=jax.numpy.uint32)
        keys = jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
            sample_indices
        )
        latent = jax.vmap(lambda key: jax.random.normal(key, model.dim))(keys)
        states = solve_ode_batched(
            model.rhs,
            latent,
            None,
            N_steps_max=steps,
            method=method,
            **dict(solver_kwargs),
        )
        images = encoder.decode(states)
        yield model_to_evaluation(np.asarray(images))
