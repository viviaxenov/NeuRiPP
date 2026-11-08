"""Native NNX time-conditioned U-Net for the acoustic transport ODE."""

from math import isqrt
from operator import index

import jax
import jax.numpy as jnp
from flax import nnx


class _TimeResBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs):
        self.norm1 = nnx.GroupNorm(in_channels, num_groups=8, rngs=rngs)
        self.conv1 = nnx.Conv(in_channels, out_channels, (3, 3), rngs=rngs)
        self.time_proj = nnx.Linear(128, out_channels, rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_channels, num_groups=8, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, (3, 3), rngs=rngs)
        self.shortcut = (
            nnx.Conv(in_channels, out_channels, (1, 1), rngs=rngs)
            if in_channels != out_channels else None
        )

    def __call__(self, x, embedding):
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        h = h + self.time_proj(embedding)[None, None, None, :]
        h = self.conv2(jax.nn.silu(self.norm2(h)))
        return h + (x if self.shortcut is None else self.shortcut(x))


class AcousticUNetRHS(nnx.Module):
    """Residual U-Net with 16/32/64 channels and scalar transport time.

    ``dim`` is the flattened square-grid size (65536 for the benchmark).
    Smaller square grids divisible by four are useful for smoke tests. Inputs
    are ``(dim,)`` or ``(B, dim)``; computations use channels-last internally.
    The deterministic fallback seed is zero; pass ``nnx.Rngs`` to override it.
    """

    def __init__(self, dim, rngs=None):
        self.dim = index(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.side = isqrt(self.dim)
        if self.side ** 2 != self.dim or self.side % 4:
            raise ValueError("dim must describe a square grid with side divisible by 4")
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.time_in = nnx.Linear(64, 128, rngs=rngs)
        self.time_out = nnx.Linear(128, 128, rngs=rngs)
        self.input_conv = nnx.Conv(1, 16, (3, 3), rngs=rngs)
        self.encoder = nnx.List([
            _TimeResBlock(c, c, rngs) for c in (16, 32, 64)
        ])
        self.downsample = nnx.List([
            nnx.Conv(a, b, (3, 3), strides=(2, 2),
                     padding=((1, 1), (1, 1)), rngs=rngs)
            for a, b in ((16, 32), (32, 64))
        ])
        self.bottleneck = nnx.List([_TimeResBlock(64, 64, rngs) for _ in range(2)])
        self.decoder = nnx.List([
            _TimeResBlock(a, b, rngs) for a, b in ((128, 64), (96, 32), (48, 16))
        ])
        self.output_conv = nnx.Conv(16, 1, (3, 3), rngs=rngs)

    def __call__(self, t, x, *args, rngs=None):
        x = jnp.asarray(x)
        t = jnp.asarray(t, dtype=x.dtype)
        if x.ndim not in (1, 2) or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},) or (B, {self.dim})")
        if t.ndim != 0:
            raise ValueError("t must be a scalar transport time")
        # Standard sinusoidal embedding, with frequencies from 1 to 1/10000.
        frequencies = jnp.exp(-jnp.log(10000.0) * jnp.arange(32) / 31)
        angles = t * frequencies
        embedding = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)))
        embedding = self.time_out(jax.nn.silu(self.time_in(embedding)))
        h = self.input_conv(x.reshape(-1, self.side, self.side, 1))
        skips = []
        for i, block in enumerate(self.encoder):
            h = block(h, embedding)
            skips.append(h)
            if i < 2:
                h = self.downsample[i](h)
        for block in self.bottleneck:
            h = block(h, embedding)
        for i, (block, skip) in enumerate(zip(self.decoder, reversed(skips))):
            if i:
                h = jnp.repeat(jnp.repeat(h, 2, axis=1), 2, axis=2)
            h = block(jnp.concatenate((h, skip), axis=-1), embedding)
        return self.output_conv(h).reshape(x.shape)


class _GuidedResBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs, *, attention=False):
        self.norm1 = nnx.GroupNorm(in_channels, num_groups=32, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_channels, out_channels, (3, 3), padding="SAME", rngs=rngs
        )
        self.time_proj = nnx.Linear(256, out_channels, rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_channels, num_groups=32, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_channels, out_channels, (3, 3), padding="SAME", rngs=rngs
        )
        self.shortcut = (
            nnx.Conv(in_channels, out_channels, (1, 1), rngs=rngs)
            if in_channels != out_channels else None
        )
        self.attention = (
            _SpatialAttention(out_channels, rngs) if attention else None
        )

    def __call__(self, x, embedding):
        h = self.conv1(jax.nn.silu(self.norm1(x)))
        h = h + self.time_proj(embedding)[None, None, None, :]
        h = self.conv2(jax.nn.silu(self.norm2(h)))
        h = h + (x if self.shortcut is None else self.shortcut(x))
        return h if self.attention is None else self.attention(h)


class _SpatialAttention(nnx.Module):
    def __init__(self, channels, rngs, *, num_heads=4):
        if channels % num_heads:
            raise ValueError("attention channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nnx.GroupNorm(channels, num_groups=32, rngs=rngs)
        self.qkv = nnx.Conv(channels, 3 * channels, (1, 1), rngs=rngs)
        self.projection = nnx.Conv(channels, channels, (1, 1), rngs=rngs)

    def __call__(self, x):
        batch, height, width, _ = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            batch, height * width, 3, self.num_heads, self.head_dim
        )
        q, k, v = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        weights = jnp.einsum("bhid,bhjd->bhij", q, k)
        weights = weights * (self.head_dim ** -0.5)
        weights = jax.nn.softmax(weights, axis=-1)
        attended = jnp.einsum("bhij,bhjd->bhid", weights, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(
            batch, height, width, self.channels
        )
        return x + self.projection(attended)


class AcousticGuidedUNetRHS(nnx.Module):
    """Literature-scale timestep-conditioned U-Net for acoustic transport.

    This NNX implementation follows the reported SFM-FWI configuration:
    base width 64, channel multipliers (1, 1, 2, 3, 4), four residual blocks
    per encoder level, and five skip-consuming decoder blocks per level. The
    paper does not specify attention placement; here attention is used at the
    penultimate width (192 channels) and in the bottleneck, which gives
    34,697,025 parameters and is documented as an implementation choice.

    ``dim`` is the flattened square-grid size. The input side must be divisible
    by 16 for four downsampling stages. Inputs may be flat or batched-flat.
    """

    widths = (64, 64, 128, 192, 256)
    blocks_per_level = 4

    def __init__(self, dim, rngs=None):
        self.dim = index(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        self.side = isqrt(self.dim)
        if self.side ** 2 != self.dim or self.side % 16:
            raise ValueError(
                "dim must describe a square grid with side divisible by 16"
            )
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.time_in = nnx.Linear(64, 256, rngs=rngs)
        self.time_out = nnx.Linear(256, 256, rngs=rngs)
        self.input_conv = nnx.Conv(1, self.widths[0], (3, 3), rngs=rngs)

        encoder = []
        downsample = []
        skip_channels = [self.widths[0]]
        current_channels = self.widths[0]
        for level, out_channels in enumerate(self.widths):
            blocks = []
            for _ in range(self.blocks_per_level):
                blocks.append(
                    _GuidedResBlock(
                        current_channels,
                        out_channels,
                        rngs,
                        attention=(level == 3),
                    )
                )
                current_channels = out_channels
                skip_channels.append(out_channels)
            encoder.append(nnx.List(blocks))
            if level < len(self.widths) - 1:
                downsample.append(
                    nnx.Conv(
                        out_channels,
                        out_channels,
                        (3, 3),
                        strides=(2, 2),
                        padding="SAME",
                        rngs=rngs,
                    )
                )
                skip_channels.append(out_channels)
        self.encoder = nnx.List(encoder)
        self.downsample = nnx.List(downsample)

        bottleneck_width = self.widths[-1]
        self.middle1 = _GuidedResBlock(bottleneck_width, bottleneck_width, rngs)
        self.middle_attention = _SpatialAttention(bottleneck_width, rngs)
        self.middle2 = _GuidedResBlock(bottleneck_width, bottleneck_width, rngs)

        decoder = []
        current_channels = bottleneck_width
        for level in reversed(range(len(self.widths))):
            out_channels = self.widths[level]
            blocks = []
            for _ in range(self.blocks_per_level + 1):
                skip_width = skip_channels.pop()
                blocks.append(
                    _GuidedResBlock(
                        current_channels + skip_width,
                        out_channels,
                        rngs,
                        attention=(level == 3),
                    )
                )
                current_channels = out_channels
            decoder.append(nnx.List(blocks))
        if skip_channels:
            raise AssertionError("U-Net skip-channel construction is unbalanced")
        self.decoder = nnx.List(decoder)
        self.upsample = nnx.List([
            nnx.Conv(width, width, (3, 3), padding="SAME", rngs=rngs)
            for width in self.widths[1:]
        ])
        self.output_norm = nnx.GroupNorm(self.widths[0], num_groups=32, rngs=rngs)
        self.output_conv = nnx.Conv(self.widths[0], 1, (3, 3), rngs=rngs)

    def __call__(self, t, x, *args, rngs=None):
        x = jnp.asarray(x)
        t = jnp.asarray(t, dtype=x.dtype)
        if x.ndim not in (1, 2) or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},) or (B, {self.dim})")
        if t.ndim != 0:
            raise ValueError("t must be a scalar transport time")
        frequencies = jnp.exp(-jnp.log(10000.0) * jnp.arange(32) / 31)
        angles = t * frequencies
        embedding = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)))
        embedding = self.time_out(jax.nn.silu(self.time_in(embedding)))

        h = self.input_conv(x.reshape(-1, self.side, self.side, 1))
        skips = [h]
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                h = block(h, embedding)
                skips.append(h)
            if level < len(self.downsample):
                h = self.downsample[level](h)
                skips.append(h)

        h = self.middle1(h, embedding)
        h = self.middle_attention(h)
        h = self.middle2(h, embedding)

        for level, blocks in zip(reversed(range(len(self.widths))), self.decoder):
            for block in blocks:
                h = block(jnp.concatenate((h, skips.pop()), axis=-1), embedding)
            if level:
                h = jnp.repeat(jnp.repeat(h, 2, axis=1), 2, axis=2)
                h = self.upsample[level - 1](h)
        h = self.output_conv(jax.nn.silu(self.output_norm(h)))
        return h.reshape(x.shape)
