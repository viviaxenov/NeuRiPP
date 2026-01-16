import math
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
    def __init__(self, din: int, rngs=None, with_counter: bool = True):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.lin = nnx.Linear(din, din, use_bias=False, rngs=rngs)
        self.counter = Counter() if with_counter else None

    def mat(self):
        return self.lin.kernel.value.T

    def __call__(self, t, x):
        if self.counter is not None:
            jax.debug.callback(lambda: self.counter.increment())

        return self.lin(x)

    def true_solution(self, t, x_init):
        A = self.mat()
        return jax.scipy.linalg.expm(A * t) @ x_init
