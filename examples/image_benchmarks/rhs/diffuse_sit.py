"""Unconditional adapter around the pinned DiffuseNNX continuous-time DiT."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import nnx

from image_benchmarks.assets.diffuse_nnx import import_diffuse_module


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


def _strip_rng_state(module: nnx.Module) -> None:
    """Neutralize non-parameter graph nodes so nnx.vmap can trace the DiT.

    The upstream DiT stores ``nnx.Rngs``/``RngStream``/``RngCount`` nodes
    (e.g. ``self.rngs`` and the forked dropout streams inside every
    ``nnx.Dropout``) and a frozen sincos table as an ``nnx.Buffer``.  The
    flow-matching trainer jits the training step and vmapped the RHS over the
    batch, and ``nnx.vmap`` refuses to extract a graph node that was already
    traced at the outer ``nnx.jit`` level ("Cannot extract graph node from
    different trace level").  Dropout is disabled for these unconditional
    models and the positional buffer never receives gradients, so both kinds
    of nodes can be demoted to static values.

    The only dropout that exists in the upstream DiT is class-label dropout
    (``ClassEmbedder``) plus block ``nnx.Dropout`` layers at rate 0.  These
    runs are unconditional (``num_classes=1`` + ``ZeroClassEmbedder``,
    ``enable_dropout=False``, ``class_dropout_prob=1.0``,
    ``mlp_dropout=attn_dropout=0.0``), so neither can ever fire and stripping
    the nodes is exact.

    Caveat: if class conditioning or block dropout is ever enabled in a DiT
    RHS, this stripping must be removed/reworked.  RNG streams then have to
    be threaded through ``nnx.vmap`` (e.g. via the explicit-dropout-key path
    ``uses_explicit_dropout_rng`` in
    ``neuripp/parametric_pushforward/flow_matching.py``) instead of being
    dropped from the graph.
    """

    from flax.nnx import rnglib

    rng_types = (rnglib.Rngs, rnglib.RngStream, rnglib.RngCount)
    for name, value in list(vars(module).items()):
        if isinstance(value, rng_types):
            setattr(module, name, nnx.data(None))
        elif isinstance(value, nnx.Variable) and not isinstance(value, nnx.Param):
            setattr(module, name, jnp.asarray(value.value))
        elif isinstance(value, nnx.List):
            for item in value:
                if isinstance(item, nnx.Module):
                    _strip_rng_state(item)
        elif isinstance(value, nnx.Module):
            _strip_rng_state(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, nnx.Module):
                    _strip_rng_state(item)


def load_diffuse_sit(
    *,
    state_shape: tuple[int, int, int],
    variant: str,
    patch_size: int,
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
    module = import_diffuse_module("diffuse_nnx.networks.transformers.dit_nnx")

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
    _strip_rng_state(model)
    return DiffuseSiTRHS(model, state_shape)
