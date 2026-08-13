"""RHS construction, presets, compatibility checks, and parameter counts."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from image_benchmarks.rhs.diffuse_sit import load_diffuse_sit
from image_benchmarks.rhs.mlp import FlattenedRHS, TimeMLP
from image_benchmarks.rhs.unet import ImageUNet, UNET_PRESETS


RHS_TYPES = {"mlp", "unet", "sit"}
DTYPES = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
}


def validate_rhs_compatibility(
    config: dict[str, Any], state_shape: tuple[int, ...]
) -> None:
    rhs_type = config.get("type")
    if rhs_type not in RHS_TYPES:
        supported = ", ".join(sorted(RHS_TYPES))
        raise ValueError(f"Unknown RHS type {rhs_type!r}; expected one of: {supported}")
    if rhs_type == "mlp":
        if len(state_shape) != 1 and not config.get("flatten", False):
            raise ValueError(
                "MLP requires a vector state unless rhs.flatten is explicitly enabled"
            )
        return
    if len(state_shape) != 3:
        raise ValueError(f"{rhs_type} requires a spatial (H, W, C) state")
    if rhs_type == "sit":
        patch_size = config.get("patch_size", 2)
        if not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("rhs.patch_size must be a positive integer")
        if state_shape[0] % patch_size or state_shape[1] % patch_size:
            raise ValueError("SiT spatial dimensions must be divisible by patch_size")
        if config.get("class_conditioning", False):
            raise ValueError("Image benchmarks are unconditional; class_conditioning must be false")


def _dtype(config: dict[str, Any]):
    name = config.get("compute_dtype", "float32")
    try:
        return DTYPES[name]
    except KeyError as error:
        raise ValueError(f"Unknown compute_dtype {name!r}") from error


def build_rhs(
    config: dict[str, Any], state_shape: tuple[int, ...], *, rngs: nnx.Rngs
):
    validate_rhs_compatibility(config, state_shape)
    rhs_type = config["type"]
    if rhs_type == "mlp":
        state_dim = math.prod(state_shape)
        time_config = config.get("time_embedding", {})
        if time_config.get("type", "sinusoidal") != "sinusoidal":
            raise ValueError("Only sinusoidal MLP time embeddings are supported")
        rhs = TimeMLP(
            state_dim,
            tuple(config.get("hidden_dims", (512, 512, 512))),
            time_embedding_dim=int(time_config.get("dim", 128)),
            activation=config.get("activation", "silu"),
            residual=bool(config.get("residual", False)),
            dtype=_dtype(config),
            rngs=rngs,
        )
        return FlattenedRHS(rhs, state_shape) if len(state_shape) != 1 else rhs
    if rhs_type == "unet":
        variant = config.get("variant", "small")
        if variant not in UNET_PRESETS:
            supported = ", ".join(sorted(UNET_PRESETS))
            raise ValueError(f"Unknown U-Net variant {variant!r}; expected: {supported}")
        preset = UNET_PRESETS[variant]
        values = {
            "base_channels": preset.base_channels,
            "channel_mult": preset.channel_mult,
            "num_res_blocks": preset.num_res_blocks,
            "attention_resolutions": preset.attention_resolutions,
            "dropout": preset.dropout,
            "num_heads": preset.num_heads,
            "num_head_channels": preset.num_head_channels,
        }
        values.update(
            {
                key: config[key]
                for key in values
                if key in config
            }
        )
        values["channel_mult"] = tuple(values["channel_mult"])
        values["attention_resolutions"] = tuple(values["attention_resolutions"])
        return ImageUNet(
            state_shape,
            **values,
            time_embedding_dim=config.get("time_embedding_dim"),
            dtype=_dtype(config),
            rngs=rngs,
        )
    if config.get("implementation", "diffuse_nnx") != "diffuse_nnx":
        raise ValueError("Only rhs.implementation='diffuse_nnx' is supported for SiT")
    return load_diffuse_sit(
        state_shape=state_shape,
        variant=config.get("variant", "S").upper(),
        patch_size=int(config.get("patch_size", 2)),
        rngs=rngs,
        dtype=_dtype(config),
    )


def parameter_count(model: nnx.Module) -> int:
    return sum(
        int(value.size)
        for value in jax.tree.leaves(nnx.state(model, nnx.Param))
    )
