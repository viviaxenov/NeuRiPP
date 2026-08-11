"""The project's small vector autoencoder, packaged for reusable adapters."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class Encoder(nnx.Module):
    def __init__(self, input_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(input_dim, latent_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return nnx.relu(self.linear(inputs))


class Decoder(nnx.Module):
    def __init__(self, latent_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(latent_dim, output_dim, rngs=rngs)

    def __call__(self, latent: jax.Array) -> jax.Array:
        return nnx.sigmoid(self.linear(latent))


class AutoEncoder(nnx.Module):
    """Single-layer vector AE retained from the existing image experiments."""

    def __init__(
        self, image_shape: tuple[int, ...], latent_dim: int, rngs: nnx.Rngs
    ):
        self.shape_in = tuple(image_shape)
        self.din = math.prod(image_shape)
        self.encoder = Encoder(self.din, latent_dim, rngs=rngs)
        self.decoder = Decoder(latent_dim, self.din, rngs=rngs)

    def __call__(self, images: jax.Array) -> jax.Array:
        return self.decode(self.encode(images))

    def encode(self, images: jax.Array) -> jax.Array:
        return self.encoder(images.reshape(images.shape[0], -1))

    def decode(self, latent: jax.Array) -> jax.Array:
        reconstruction = self.decoder(latent)
        return reconstruction.reshape(latent.shape[0], *self.shape_in)
