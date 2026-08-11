"""Time-conditioned vector MLP."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from neuripp.image_benchmarks.rhs.common import sinusoidal_time_embedding


ACTIVATIONS: dict[str, Callable] = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "selu": jax.nn.selu,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "tanh": jnp.tanh,
}


class TimeMLP(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        time_embedding_dim: int,
        activation: str,
        residual: bool,
        dtype,
        rngs: nnx.Rngs,
    ):
        if state_dim < 1 or not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("MLP state and hidden dimensions must be positive")
        if activation not in ACTIVATIONS:
            supported = ", ".join(sorted(ACTIVATIONS))
            raise ValueError(f"Unknown MLP activation {activation!r}; expected: {supported}")
        self.dim = state_dim
        self.time_embedding_dim = int(time_embedding_dim)
        self.activation = ACTIVATIONS[activation]
        self.residual = bool(residual)
        self.dtype = dtype
        widths = (state_dim + self.time_embedding_dim, *hidden_dims, state_dim)
        self.layers = nnx.List(
            [
                nnx.Linear(
                    widths[index], widths[index + 1], dtype=dtype, rngs=rngs
                )
                for index in range(len(widths) - 1)
            ]
        )

    def __call__(self, time, state, *args):
        del args
        if state.shape != (self.dim,):
            raise ValueError(f"MLP expected state shape ({self.dim},), got {state.shape}")
        time_embedding = sinusoidal_time_embedding(
            time, self.time_embedding_dim
        ).astype(self.dtype)
        hidden = jnp.concatenate((state.astype(self.dtype), time_embedding), axis=-1)
        for layer in self.layers[:-1]:
            transformed = self.activation(layer(hidden))
            if self.residual and transformed.shape == hidden.shape:
                hidden = hidden + transformed
            else:
                hidden = transformed
        return self.layers[-1](hidden)


class FlattenedRHS(nnx.Module):
    """Explicit flatten/unflatten boundary for controlled image experiments."""

    def __init__(self, rhs: TimeMLP, state_shape: tuple[int, ...]):
        self.rhs = rhs
        self.dim = tuple(state_shape)

    def __call__(self, time, state, *args):
        prediction = self.rhs(time, state.reshape(-1), *args)
        return prediction.reshape(self.dim)
