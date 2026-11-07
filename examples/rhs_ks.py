"""NNX equivalent of the benchmark PDEquinox DilatedResNet (no dependency).

Conventions follow pdequinox/arch/_dilated_res_net.py and
pdequinox/blocks/_dilated_res_block.py: linear pointwise lifting/projection,
pre-convolution one-group normalization, post-convolution activation, and an
identity residual bypass. Convolution weights and biases use the Equinox
uniform initialization with bounds +/- 1/sqrt(fan_in).
"""

from operator import index

import jax
import jax.numpy as jnp
from flax import nnx


def _conv(in_channels, out_channels, kernel_size, rngs, dilation=1):
    bound = (in_channels * kernel_size) ** -0.5

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return nnx.Conv(
        in_channels, out_channels, (kernel_size,),
        kernel_dilation=(dilation,), padding="CIRCULAR",
        kernel_init=init, bias_init=init, rngs=rngs,
    )


class _DilatedResBlock(nnx.Module):
    def __init__(self, channels, dilation_rates, rngs):
        self.norms = nnx.List([
            nnx.GroupNorm(channels, num_groups=1, epsilon=1e-5,
                          use_fast_variance=False, rngs=rngs)
            for _ in dilation_rates
        ])
        self.convs = nnx.List([
            _conv(channels, channels, 3, rngs, d) for d in dilation_rates
        ])

    def __call__(self, x):
        h = x
        for norm, conv in zip(self.norms, self.convs):
            h = jax.nn.silu(conv(norm(h)))
        return x + h


class KSDilatedResNetRHS(nnx.Module):
    """Periodic scalar-time RHS preserving flat or batched-flat input shape.

    Defaults reproduce the 111,073-parameter architecture on a ``dim=200``
    grid. Coordinate channels break translation invariance, as needed for a
    posterior with fixed sensors. No random state is used during evaluation.
    """

    def __init__(self, dim, rngs=None, *, hidden_channels=32, num_blocks=5,
                 dilation_rates=(1, 2, 4, 8, 4, 2, 1)):
        self.dim = index(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if hidden_channels < 1 or num_blocks < 1:
            raise ValueError("hidden_channels and num_blocks must be positive")
        dilation_rates = tuple(index(d) for d in dilation_rates)
        if not dilation_rates or any(d < 1 for d in dilation_rates):
            raise ValueError("dilation_rates must contain positive integers")
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.lifting = _conv(4, hidden_channels, 1, rngs)
        self.blocks = nnx.List([
            _DilatedResBlock(hidden_channels, dilation_rates, rngs)
            for _ in range(num_blocks)
        ])
        self.projection = _conv(hidden_channels, 1, 1, rngs)

    def __call__(self, t, x, *args, rngs=None):
        x = jnp.asarray(x)
        t = jnp.asarray(t, dtype=x.dtype)
        if x.ndim not in (1, 2) or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},) or (B, {self.dim})")
        if t.ndim != 0:
            raise ValueError("t must be a scalar transport time")
        h = x.reshape(-1, self.dim, 1)
        angle = 2 * jnp.pi * jnp.arange(self.dim, dtype=x.dtype) / self.dim
        channels = [h, jnp.broadcast_to(t, h.shape)]
        channels.extend(jnp.broadcast_to(c[None, :, None], h.shape)
                        for c in (jnp.sin(angle), jnp.cos(angle)))
        h = self.lifting(jnp.concatenate(channels, axis=-1))
        for block in self.blocks:
            h = block(h)
        return self.projection(h).reshape(x.shape)
