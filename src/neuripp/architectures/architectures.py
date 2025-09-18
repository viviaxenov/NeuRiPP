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


# MLP
class MLP(nnx.Module):
    def __init__(
        self,
        din: int,
        num_layers: int,
        width_layers: int,
        dout: int,
        activation_fn: str,
        rngs: nnx.Rngs,
    ):

        activation_fn = str_to_act_fn(activation_fn)

        layers = []

        in_dim = din

        # hidden layers
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

        # output layer (no activation)
        layers.append(nnx.Linear(in_dim, dout, rngs=rngs))

        self.layers = layers

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(nnx.Module):
    def __init__(
        self,
        din: int,
        num_layers: int,
        width_layers: int,
        dout: int,
        activation_fn: str,
        rngs: nnx.Rngs,
    ):

        activation_fn = str_to_act_fn(activation_fn)

        layers = []

        in_dim = din

        # hidden layers
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

        # output layer (no activation)
        layers.append(nnx.Linear(in_dim, dout, rngs=rngs))

        self.layers = layers

    def __call__(self, x: Array) -> Array:
        x_copy = x.copy()
        for layer in self.layers:
            x = layer(x)
        return x_copy + x
