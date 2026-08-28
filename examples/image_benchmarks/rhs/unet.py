"""Flax NNX U-Net following the TorchCFM/OpenAI architecture semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from image_benchmarks.rhs.common import (
    group_count,
    sinusoidal_time_embedding,
)


@dataclass(frozen=True)
class UNetPreset:
    base_channels: int
    channel_mult: tuple[int, ...]
    num_res_blocks: int
    attention_resolutions: tuple[int, ...]
    dropout: float
    num_heads: int = 1
    num_head_channels: int | None = None


UNET_PRESETS = {
    "small": UNetPreset(64, (1, 2, 2, 2), 2, (16,), 0.0),
    "cifar_reference": UNetPreset(
        128, (1, 2, 2, 2), 2, (16,), 0.1, 4, 64
    ),
    "large": UNetPreset(192, (1, 2, 3, 4), 2, (16, 8), 0.1),
}


class ResBlock(nnx.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_dim: int,
        dropout: float,
        *,
        rngs: nnx.Rngs,
        dtype: Any = jnp.float32,
    ):
        self.norm1 = nnx.GroupNorm(
            input_channels,
            num_groups=group_count(input_channels),
            dtype=dtype,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            input_channels,
            output_channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=dtype,
            rngs=rngs,
        )
        self.time_projection = nnx.Linear(
            time_dim, output_channels, dtype=dtype, rngs=rngs
        )
        self.norm2 = nnx.GroupNorm(
            output_channels,
            num_groups=group_count(output_channels),
            dtype=dtype,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout)
        self.conv2 = nnx.Conv(
            output_channels,
            output_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            rngs=rngs,
        )
        self.shortcut = (
            nnx.Conv(
                input_channels,
                output_channels,
                kernel_size=(1, 1),
                dtype=dtype,
                rngs=rngs,
            )
            if input_channels != output_channels
            else None
        )

    def __call__(self, features, time_embedding, *, rngs=None):
        residual = features if self.shortcut is None else self.shortcut(features)
        hidden = self.conv1(jax.nn.silu(self.norm1(features)))
        hidden = hidden + self.time_projection(jax.nn.silu(time_embedding))[None, None, :]
        hidden = self.conv2(
            self.dropout(jax.nn.silu(self.norm2(hidden)), rngs=rngs)
        )
        return residual + hidden


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        num_head_channels: int | None,
        rngs: nnx.Rngs,
        dtype: Any = jnp.float32,
    ):
        heads = (
            channels // num_head_channels
            if num_head_channels is not None and channels % num_head_channels == 0
            else num_heads
        )
        if num_head_channels is not None and channels % num_head_channels:
            raise ValueError(
                f"Attention channels {channels} are not divisible by "
                f"num_head_channels={num_head_channels}"
            )
        if heads < 1 or channels % heads:
            raise ValueError(
                f"Attention channels {channels} are not divisible by {heads} heads"
            )
        self.norm = nnx.GroupNorm(
            channels, num_groups=group_count(channels), dtype=dtype, rngs=rngs
        )
        self.attention = nnx.MultiHeadAttention(
            heads,
            channels,
            out_kernel_init=nnx.initializers.zeros_init(),
            decode=False,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, features):
        height, width, channels = features.shape
        sequence = self.norm(features).reshape(1, height * width, channels)
        attended = self.attention(sequence).reshape(height, width, channels)
        return features + attended


class SpatialBlock(nnx.Module):
    def __init__(self, residual: ResBlock, attention: AttentionBlock | None):
        self.residual = residual
        self.attention = attention

    def __call__(self, features, time_embedding, *, rngs=None):
        features = self.residual(features, time_embedding, rngs=rngs)
        if self.attention is not None:
            features = self.attention(features)
        return features


class Downsample(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs, dtype: Any):
        self.conv = nnx.Conv(
            channels,
            channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, features):
        return self.conv(features)


class Upsample(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs, dtype: Any):
        self.conv = nnx.Conv(
            channels,
            channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, features):
        height, width, channels = features.shape
        features = jax.image.resize(
            features,
            (height * 2, width * 2, channels),
            method="nearest",
        )
        return self.conv(features)


class DownStage(nnx.Module):
    def __init__(self, blocks, downsample):
        self.blocks = nnx.List(blocks)
        self.downsample = downsample


class UpStage(nnx.Module):
    def __init__(self, blocks, upsample):
        self.blocks = nnx.List(blocks)
        self.upsample = upsample


class ImageUNet(nnx.Module):
    """Unconditional channels-last U-Net for one spatial FM state."""

    def __init__(
        self,
        state_shape: tuple[int, int, int],
        *,
        base_channels: int,
        channel_mult: tuple[int, ...],
        num_res_blocks: int,
        attention_resolutions: tuple[int, ...],
        dropout: float,
        num_heads: int = 1,
        num_head_channels: int | None = None,
        time_embedding_dim: int | None = None,
        dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if len(state_shape) != 3:
            raise ValueError("U-Net state must have shape (H, W, C)")
        if not channel_mult or num_res_blocks < 1 or base_channels < 1:
            raise ValueError("Invalid U-Net channel or residual-block configuration")
        self.dim = tuple(state_shape)
        self.time_frequency_dim = base_channels
        time_dim = time_embedding_dim or base_channels * 4
        self.time_mlp = nnx.Sequential(
            nnx.Linear(base_channels, time_dim, dtype=dtype, rngs=rngs),
            nnx.silu,
            nnx.Linear(time_dim, time_dim, dtype=dtype, rngs=rngs),
        )
        input_channels = state_shape[-1]
        channels = base_channels * channel_mult[0]
        self.input_conv = nnx.Conv(
            input_channels,
            channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=dtype,
            rngs=rngs,
        )
        skip_channels = [channels]
        down_stages = []
        resolution = state_shape[0]
        for level, multiplier in enumerate(channel_mult):
            output_channels = base_channels * multiplier
            blocks = []
            for _ in range(num_res_blocks):
                residual = ResBlock(
                    channels,
                    output_channels,
                    time_dim,
                    dropout,
                    rngs=rngs,
                    dtype=dtype,
                )
                channels = output_channels
                attention = (
                    AttentionBlock(
                        channels,
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        rngs=rngs,
                        dtype=dtype,
                    )
                    if resolution in attention_resolutions
                    else None
                )
                blocks.append(SpatialBlock(residual, attention))
                skip_channels.append(channels)
            downsample = None
            if level != len(channel_mult) - 1:
                downsample = Downsample(channels, rngs=rngs, dtype=dtype)
                skip_channels.append(channels)
                resolution = (resolution + 1) // 2
            down_stages.append(DownStage(blocks, downsample))
        self.down_stages = nnx.List(down_stages)

        self.middle1 = ResBlock(
            channels, channels, time_dim, dropout, rngs=rngs, dtype=dtype
        )
        self.middle_attention = AttentionBlock(
            channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            rngs=rngs,
            dtype=dtype,
        )
        self.middle2 = ResBlock(
            channels, channels, time_dim, dropout, rngs=rngs, dtype=dtype
        )

        up_stages = []
        for level in reversed(range(len(channel_mult))):
            output_channels = base_channels * channel_mult[level]
            blocks = []
            for _ in range(num_res_blocks + 1):
                skip_channels_for_block = skip_channels.pop()
                residual = ResBlock(
                    channels + skip_channels_for_block,
                    output_channels,
                    time_dim,
                    dropout,
                    rngs=rngs,
                    dtype=dtype,
                )
                channels = output_channels
                attention = (
                    AttentionBlock(
                        channels,
                        num_heads=num_heads,
                        num_head_channels=num_head_channels,
                        rngs=rngs,
                        dtype=dtype,
                    )
                    if resolution in attention_resolutions
                    else None
                )
                blocks.append(SpatialBlock(residual, attention))
            upsample = None
            if level > 0:
                upsample = Upsample(channels, rngs=rngs, dtype=dtype)
                resolution *= 2
            up_stages.append(UpStage(blocks, upsample))
        if skip_channels:
            raise AssertionError("U-Net skip-channel construction did not balance")
        self.up_stages = nnx.List(up_stages)
        self.output_norm = nnx.GroupNorm(
            channels, num_groups=group_count(channels), dtype=dtype, rngs=rngs
        )
        self.output_conv = nnx.Conv(
            channels,
            input_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
            rngs=rngs,
        )

    @staticmethod
    def _resize_like(features, reference):
        if features.shape[:2] == reference.shape[:2]:
            return features
        return jax.image.resize(
            features,
            (*reference.shape[:2], features.shape[-1]),
            method="nearest",
        )

    def __call__(self, time, state, *args, rngs=None):
        if state.shape != self.dim:
            raise ValueError(f"U-Net expected state shape {self.dim}, got {state.shape}")
        time_embedding = self.time_mlp(
            sinusoidal_time_embedding(time, self.time_frequency_dim)
        )
        features = self.input_conv(state)
        skips = [features]
        for stage in self.down_stages:
            for block in stage.blocks:
                features = block(features, time_embedding, rngs=rngs)
                skips.append(features)
            if stage.downsample is not None:
                features = stage.downsample(features)
                skips.append(features)
        features = self.middle1(features, time_embedding, rngs=rngs)
        features = self.middle_attention(features)
        features = self.middle2(features, time_embedding, rngs=rngs)
        for stage in self.up_stages:
            for block in stage.blocks:
                skip = skips.pop()
                features = self._resize_like(features, skip)
                features = block(
                    jnp.concatenate((features, skip), axis=-1),
                    time_embedding,
                    rngs=rngs,
                )
            if stage.upsample is not None:
                features = stage.upsample(features)
        if skips:
            raise AssertionError("U-Net forward did not consume all skip features")
        features = self._resize_like(features, state)
        return self.output_conv(jax.nn.silu(self.output_norm(features)))
