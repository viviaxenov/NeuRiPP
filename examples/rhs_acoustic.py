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
