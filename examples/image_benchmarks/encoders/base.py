"""Common image representation contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EncoderAdapter(Protocol):
    is_stochastic: bool

    def encode(self, images: Any, rng: Any | None = None) -> Any: ...

    def decode(self, latent: Any, rng: Any | None = None) -> Any: ...

    def latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]: ...
