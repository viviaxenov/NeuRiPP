"""Pass-through pixel representation."""

from __future__ import annotations

from typing import Any


class IdentityEncoder:
    is_stochastic = False
    checkpoint_sha256 = None

    def encode(self, images: Any, rng: Any | None = None) -> Any:
        del rng
        return images

    def decode(self, latent: Any, rng: Any | None = None) -> Any:
        del rng
        return latent

    def latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(input_shape)
