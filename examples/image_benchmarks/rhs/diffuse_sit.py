"""Unconditional adapter around the pinned DiffuseNNX continuous-time DiT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
from flax import nnx

from image_benchmarks.assets.diffuse_nnx import (
    import_diffuse_module,
    prepare_diffuse_nnx_source,
)


SIT_VARIANTS = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}


class DiffuseSiTRHS(nnx.Module):
    def __init__(self, model: Any, state_shape: tuple[int, int, int]):
        self.model = model
        self.dim = tuple(state_shape)

    def __call__(self, time, state, *args):
        del args
        prediction = self.model(
            state[None, ...], jnp.asarray(time, dtype=state.dtype).reshape(1), y=None
        )
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        return prediction[0]


class ZeroClassEmbedder(nnx.Module):
    """Parameter-free unconditional replacement for upstream ClassEmbedder."""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def __call__(self, labels):
        return jnp.zeros((labels.shape[0], self.hidden_size), dtype=jnp.float32)


def load_diffuse_sit(
    *,
    state_shape: tuple[int, int, int],
    variant: str,
    patch_size: int,
    source_dir: str | Path,
    source_auto_download: bool,
    rngs: nnx.Rngs,
    dtype=jnp.float32,
) -> DiffuseSiTRHS:
    if variant not in SIT_VARIANTS:
        supported = ", ".join(SIT_VARIANTS)
        raise ValueError(f"Unknown SiT variant {variant!r}; expected: {supported}")
    height, width, channels = state_shape
    if height != width:
        raise ValueError("DiffuseNNX DiT currently requires a square spatial state")
    if height % patch_size or width % patch_size:
        raise ValueError("SiT state dimensions must be divisible by patch_size")
    source_dir = prepare_diffuse_nnx_source(
        source_dir, auto_download=source_auto_download
    )
    module = import_diffuse_module("networks.transformers.dit_nnx", source_dir)

    class CompatibleDiT(module.DiT):
        """Flax 0.12 compatibility shim for the upstream Python block list."""

        def __setattr__(self, name, value):
            if name == "blocks" and isinstance(value, list):
                value = nnx.List(value)
            super().__setattr__(name, value)

    model = CompatibleDiT(
        input_size=height,
        patch_size=patch_size,
        in_channels=channels,
        continuous_time_embed=True,
        num_classes=1,
        class_dropout_prob=1.0,
        enable_dropout=False,
        dtype=dtype,
        rngs=rngs,
        **SIT_VARIANTS[variant],
    )
    model.y_embedder = ZeroClassEmbedder(SIT_VARIANTS[variant]["hidden_size"])
    model.num_classes = 0
    return DiffuseSiTRHS(model, state_shape)
