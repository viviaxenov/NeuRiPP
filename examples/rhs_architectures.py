import jax
import jax.numpy as jnp
from flax import nnx

from typing import Tuple, Callable

from neuripp.image_benchmarks.encoders.project_ae_model import (
    AutoEncoder,
    Decoder,
    Encoder,
)


class Count(nnx.Variable):
    pass


class Counter(nnx.Module):
    def __init__(self):
        self.count = Count(jnp.array(0))

    def increment(self):
        self.count.value += 1

    def get(self):
        return self.count.value

    def reset(self):
        self.count.value *= 0


class LinearRHS(nnx.Module):
    def __init__(self, dim: int, rngs=None, with_counter: bool = True):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.lin = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.dim = dim
        self.counter = Counter() if with_counter else None

    def mat(self):
        return self.lin.kernel.value.T

    def __call__(self, t, x, *args):
        if self.counter is not None:
            jax.debug.callback(lambda: self.counter.increment())

        return self.lin(x)

    def true_solution(self, t, x_init):
        A = self.mat()
        return jax.scipy.linalg.expm(A * t) @ x_init


class MLP(nnx.Module):
    def __init__(
        self,
        dim,
        rngs,
        n_hidden: int = 1,
        dim_hidden: int = None,
        activation: nnx.Module = nnx.selu,
    ):
        self.dim = dim
        if dim_hidden is None:
            dim_hidden = dim

        # self.t_proj = nnx.Sequential(nnx.Linear(1, dim, rngs=rngs), nnx.selu)

        list_of_modules = (
            [
                nnx.Linear(dim + 1, dim_hidden, rngs=rngs),
                activation,
            ]
            + [
                nnx.Linear(dim_hidden, dim_hidden, rngs=rngs),
                activation,
            ]
            * (n_hidden - 1)
            + [
                nnx.Linear(dim_hidden, dim, rngs=rngs),
            ]
        )

        self.mlp = nnx.Sequential(*list_of_modules)

    def __call__(self, t, x, *args):
        y = jnp.concat((x, jnp.atleast_1d(t)), axis=-1)
        return self.mlp(y)


class FFJORDConv2D(nnx.Module):
    """Convolutional model with time embedding via concatenation similar to one used in FFJORD paper"""

    def __init__(
        self,
        dim: Tuple[int],
        rngs: nnx.Rngs,
        activation_fn: Callable = nnx.swish,
        n_layers: int = 3,
        dim_hidden: int = 64,
    ):
        self.dim = dim
        list_of_shapes = (1,) + (dim_hidden,) * n_layers + (1,)
        self._conv_layers = nnx.List(
            nnx.Conv(
                list_of_shapes[i] + 1,
                list_of_shapes[i + 1],
                kernel_size=3,
                rngs=rngs,
            )
            for i in range(n_layers + 1)
        )
        self._activations = nnx.List(
                (activation_fn,) * n_layers + (lambda _x: _x,)
        )

    def __call__(self, t, x, *args):
        # reshape from flat (needed for ODE)
        # and add channel dimension
        x = x.reshape( *self.dim, 1)

        for _layer, _activation in zip(self._conv_layers, self._activations):
            # broadcast t to x shape
            # tt = jnp.broadcast_to(t[:, None, None, None], x.shape[:-1] + (1,))
            tt = jnp.full((*x.shape[:-1], 1), t)
            # concatenate t as another channel
            xtt = jnp.concatenate((x, tt), axis=-1)
            x = _activation(_layer(xtt))

        # flatten x again
        return x.ravel()


class _CNNLayer(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        channels_in: int,
        channels_out: int = None,
        n_groups: int = 8,
        gn=True,
        act=True,
        dilation=1,
    ):
        if channels_out is None:
            channels_out = channels_in
        self.gn = (
            nnx.GroupNorm(channels_out, num_groups=n_groups, rngs=rngs)
            if gn
            else lambda _x: _x
        )
        self.act = nnx.silu if gn else lambda _x: _x
        self.conv = nnx.Conv(
            channels_in,
            channels_out,
            kernel_size=(3, 3),
            input_dilation=dilation,
            rngs=rngs,
        )

    def __call__(self, x, time_embedding):
        x += time_embedding
        x += self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x, time_embedding


class CFMConv2D(nnx.Module):
    """Convolutional NN from the flow matching tutorial https://github.com/Jackson-Kang/Pytorch-Conditional-Flow-Matching-Tutorial/blob/main/01_conditional-flow-matching.ipynb"""

    def __init__(
        self,
        dim: Tuple[int],
        rngs: nnx.Rngs,
        time_embedding_dim: int = 6,
        n_channels: int = 64,
        n_hidden: int = 2,
    ):
        self.dim = dim
        self.time_embedding_dim = time_embedding_dim
        self.time_proj = nnx.Sequential(
            nnx.Linear(self.time_embedding_dim, self.time_embedding_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(self.time_embedding_dim, n_channels, rngs=rngs),
            nnx.relu,
        )
        self.in_project = nnx.Conv(1, n_channels, kernel_size=(7, 7), rngs=rngs)
        blocks = [
            _CNNLayer(  # why do we need this block w/o activation and gn???
                rngs,
                channels_in=n_channels,
                channels_out=n_channels,
                act=False,
                gn=False,
            )
        ] + [_CNNLayer(rngs, channels_in=n_channels, dilation=3**(idx//2)) for idx in range(n_hidden)] 
        self.residual_blocks = nnx.Sequential(*blocks)
        self.out_proj = nnx.Conv(n_channels, 1, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, t, x, *args, rngs=None):
        x = x.reshape((*self.dim, 1))
        t_emb = self.get_timestep_embedding(t)
        x = self.in_project(x)
        x_r = self.residual_blocks(x, t_emb, rngs=rngs)[0]
        return self.out_proj(x_r).ravel()

    def get_timestep_embedding(
        self,
        t: jnp.ndarray,
        downscale_freq_shift: "float" = 0,
        max_period: int = 10000,
    ):
        half_dim = self.time_embedding_dim // 2
        exponent = -jnp.log(max_period) * jnp.arange(
            start=0, stop=half_dim, dtype=jnp.float32
        )
        exponent = exponent / (half_dim - downscale_freq_shift)
        emb = jnp.exp(exponent)
        # emb = t[:, None] * emb[None, :]
        emb = t*emb
        emb = jnp.concat([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        return self.time_proj(emb)[None, None, :]
