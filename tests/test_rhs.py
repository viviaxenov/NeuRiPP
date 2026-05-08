import jax
import jax.numpy as jnp
from flax import nnx


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
        n_hidden: int = 1,
        dim_hidden: int = None,
        seed: int = 42,
        activation: nnx.Module = nnx.selu,
    ):
        self.rngs = nnx.Rngs(seed)
        self.dim = dim
        if dim_hidden is None:
            dim_hidden = dim

        self.t_proj = nnx.Sequential(nnx.Linear(1, dim, rngs=self.rngs), nnx.selu)

        list_of_modules = (
            [
                nnx.Linear(dim + 1, dim_hidden, rngs=self.rngs),
                activation,
            ]
            + [
                nnx.Linear(dim_hidden, dim_hidden, rngs=self.rngs),
                activation,
            ]
            * n_hidden
            + [
                nnx.Linear(dim_hidden, dim, rngs=self.rngs),
            ]
        )

        self.mlp = nnx.Sequential(*list_of_modules)

    def __call__(self, t, x, *args):
        y = jnp.concat((x, jnp.atleast_1d(t)), axis=-1)
        return self.mlp(y)
