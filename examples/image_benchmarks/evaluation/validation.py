"""Fixed held-out Flow Matching objective evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from neuripp.parametric_pushforward.flow_matching import flow_matching_loss


@dataclass(frozen=True)
class FixedFMValidationSet:
    states: np.ndarray
    times: np.ndarray
    noise: np.ndarray
    dropout_keys: np.ndarray
    identifiers: tuple[str, ...]
    seed: int

    def __post_init__(self):
        count = self.states.shape[0]
        if (
            self.noise.shape != self.states.shape
            or self.times.shape != (count,)
            or self.dropout_keys.shape != (count, 2)
        ):
            raise ValueError("Fixed FM states, noise, and times have incompatible shapes")
        if len(self.identifiers) != count:
            raise ValueError("Fixed FM identifiers do not match the state count")


def make_fixed_fm_validation(
    states: np.ndarray,
    identifiers: Sequence[str],
    seed: int,
) -> FixedFMValidationSet:
    states = np.asarray(states, dtype=np.float32)
    time_key, noise_key, dropout_key = jax.random.split(jax.random.key(seed), 3)
    times = np.asarray(jax.random.uniform(time_key, (states.shape[0],)))
    noise = np.asarray(jax.random.normal(noise_key, states.shape), dtype=np.float32)
    dropout_keys = np.asarray(
        jax.random.key_data(jax.random.split(dropout_key, states.shape[0]))
    )
    return FixedFMValidationSet(
        states=states,
        times=times,
        noise=noise,
        dropout_keys=dropout_keys,
        identifiers=tuple(str(identifier) for identifier in identifiers),
        seed=int(seed),
    )


def evaluate_fixed_fm_loss(
    model,
    validation: FixedFMValidationSet,
    *,
    batch_size: int,
) -> float:
    if batch_size < 1:
        raise ValueError("FM validation batch_size must be positive")
    weighted_loss = 0.0
    count = 0
    for start in range(0, len(validation.states), batch_size):
        stop = min(start + batch_size, len(validation.states))
        loss = flow_matching_loss(
            model,
            jnp.asarray(validation.states[start:stop]),
            None,
            times=jnp.asarray(validation.times[start:stop]),
            noise=jnp.asarray(validation.noise[start:stop]),
        )
        batch_count = stop - start
        weighted_loss += float(loss) * batch_count
        count += batch_count
    if not count:
        raise ValueError("FM validation set is empty")
    return weighted_loss / count
