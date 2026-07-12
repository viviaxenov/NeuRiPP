from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

import numpy as np
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward

import datasets


class BaseBatcher(nnx.Module, ABC):
    def __init__(self, shape: int | Tuple[int], resample_each: int):
        if isinstance(shape, int):
            shape = (shape,)
        if resample_each <= 0:
            raise ValueError("resample_each must be positive.")

        self.shape = shape
        self.resample_each = resample_each

        self.x = nnx.Variable(jnp.zeros((*shape, 2)))
        self.counter = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    @abstractmethod
    def _sample(self, rngs: nnx.Rngs) -> jax.Array:
        raise NotImplementedError

    def __call__(self, rngs: nnx.Rngs) -> jax.Array:
        do_resample = self.counter.value == 0

        def reuse_fn(*args):
            return self.x.value

        if do_resample:
            x = self._sample(rngs)
            self.x.set_value(x)

        self.counter.set_value((self.counter.value + 1) % self.resample_each)

        return self.x.value


class CheckerboardBatcher(BaseBatcher):
    def __init__(self, shape: int, resample_each: int):
        super().__init__(shape=shape, resample_each=resample_each)

    def _sample(self, rngs: nnx.Rngs) -> jax.Array:
        points = rngs.data.uniform(
            shape=(*self.shape, 2),
        )

        shifts_x = (
            rngs.data.randint(
                shape=self.shape,
                minval=0,
                maxval=4,
            )
            - 2
        )

        shifts_y = (
            rngs.data.randint(
                shape=self.shape,
                minval=0,
                maxval=2,
            )
            * 2
            + shifts_x % 2
            - 2
        )

        points = points.at[..., 0].add(shifts_x)
        points = points.at[..., 1].add(shifts_y)

        return points


class TwoSpiralsBatcher(BaseBatcher):
    def __init__(self, shape: int, resample_each: int):
        super().__init__(shape=shape, resample_each=resample_each)
        self.n_samples = nnx.Variable(np.prod(self.shape, dtype=int))

    def _sample(self, rngs: nnx.Rngs) -> jax.Array:
        half = self.n_samples.value // 2
        other_half = self.n_samples.value - half

        n = (
            jnp.sqrt(
                rngs.data.uniform(
                    shape=(half, 1),
                )
            )
            * 540.0
            * (2.0 * jnp.pi)
            / 360.0
        )

        d1x = -jnp.cos(n) * n + rngs.data.uniform(shape=(half, 1)) * 0.5
        d1y = jnp.sin(n) * n + rngs.data.uniform(shape=(half, 1)) * 0.5

        n = (
            jnp.sqrt(
                rngs.data.uniform(
                    shape=(other_half, 1),
                )
            )
            * 540.0
            * (2.0 * jnp.pi)
            / 360.0
        )
        d2x = jnp.cos(n) * n + rngs.data.uniform(shape=(other_half, 1)) * 0.5
        d2y = -jnp.sin(n) * n + rngs.data.uniform(shape=(other_half, 1)) * 0.5
        x = (
            jnp.vstack(
                (
                    jnp.hstack((d1x, d1y)),
                    jnp.hstack((d2x, d2y)),
                )
            )
            / 3.0
        )

        x = x + rngs.data.uniform(shape=x.shape) * 0.1

        return x.reshape(*self.shape, 2, order="F")


class EightGaussiansBatcher(BaseBatcher):
    def __init__(self, shape: int, resample_each: int):
        super().__init__(shape=shape, resample_each=resample_each)

        theta = jnp.linspace(
            0.0,
            2.0 * jnp.pi,
            8,
            endpoint=False,
        )

        self.centers = nnx.Variable(
            4.0
            * jnp.stack(
                (jnp.cos(theta), jnp.sin(theta)),
                axis=-1,
            )
        )

    def _sample(self, rngs: nnx.Rngs) -> jax.Array:
        blob = (
            rngs.data.normal(
                shape=(*self.shape, 2),
            )
            * 0.5
        )

        shift_ids = rngs.data.randint(
            shape=self.shape,
            minval=0,
            maxval=8,
        )

        x = blob + self.centers.value[shift_ids, :]
        x = x / 1.414

        return x


class DatasetBatcher(BaseBatcher):
    def __init__(
        self,
        shape: int | Tuple[int, ...],
        resample_each: int,
        X_train,
    ):
        super().__init__(shape=shape, resample_each=resample_each)
        self.n_samples = nnx.Variable(np.prod(self.shape, dtype=int))
        self.X_train = X_train
        self.i = nnx.Variable(-1)
        self.x = nnx.Variable(jnp.zeros((*self.shape, self.X_train.shape[-1])))

    def _sample(self, rngs: nnx.Rngs):
        batch_size = int(self.n_samples.value)
        self.i.value = (self.i.value + 1) % (self.X_train.shape[0] // batch_size)
        if self.i.value == 0:
            self.X_train = rngs.permutation(self.X_train)
        return self.X_train[
            self.i.value * batch_size : (self.i.value + 1) * batch_size, ...
        ].reshape((*self.shape, -1))


class LatentBatcherFromModel(BaseBatcher):
    def __init__(
        self, shape: int | Tuple[int, ...], resample_each: int, model: ParametricPushforward
    ):
        self._model = model
        super().__init__(shape=shape, resample_each=resample_each)
        self.n_samples = nnx.Variable(np.prod(self.shape, dtype=int))

    def _sample(self, rngs):
        return self._model._sample_latent(self.n_samples.value, rngs).reshape(
            (*self.shape, -1)
        )


class ZipBatcher(nnx.Module, pytree=False):
    def __init__(self, *batchers: Tuple[BaseBatcher]):
        self.batchers = batchers

    def __call__(self, rngs: nnx.Rngs):
        return tuple(b(rngs) for b in self.batchers)
