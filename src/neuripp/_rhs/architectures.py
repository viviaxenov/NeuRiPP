"""
In this file we include the basic architectures for the neural network.
"""

import flax.nnx as nnx
from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from jax._src import api
from jax.nn.initializers import xavier_uniform, normal, xavier_normal

# initializer = jax.nn.initializers.xavier_uniform()


# Define SinTu activation fn
@api.jit
def SinTu(x: ArrayLike) -> Array:
    return jnp.sin(jnp.maximum(x, 0.0))


# Function from string to activation function


def str_to_act_fn(name: str) -> Callable:
    if name == "relu":
        return nnx.relu
    elif name == "sigmoid":
        return nnx.sigmoid
    elif name == "tanh":
        return nnx.tanh
    elif name == "SinTu":
        return SinTu
    else:
        raise ValueError(f"Unknown activation function: {name}")




class ResNet(nnx.Module):
    def __init__(
        self,
        din: int,
        num_layers: int,
        width_layers: int,
        dout: int,
        activation_fn: str,
        rngs: nnx.Rngs,
        num_blocks: int = 1,
    ):

        activation_fn = str_to_act_fn(activation_fn)

        blocks = []
        layers = []

        in_dim = din

        # hidden layers
        for _ in range(num_blocks):
            layers = []
            for _ in range(num_layers):
                layers.append(
                    nnx.Linear(
                        in_dim,
                        width_layers,
                        rngs=rngs,
                        kernel_init=xavier_uniform(),
                        bias_init=normal(stddev=1e-3),
                    )
                )  # ,  bias_init = normal(stddev=1e-3)
                layers.append(activation_fn)
                in_dim = width_layers
            blocks.append(layers)

        # output layer (no activation)
        blocks.append([nnx.Linear(in_dim, dout, rngs=rngs)])

        self.blocks = blocks

    def __call__(self, x: Array) -> Array:
        for layers in self.blocks:
            x_skip = x.copy()
            for layer in layers:
                x = layer(x)
            x = x_skip + x

        return x


