from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx

from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward


class BaseBatcher(nnx.Module, ABC):
    def __init__(self, n_samples: int, resample_each: int):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if resample_each <= 0:
            raise ValueError("resample_each must be positive.")

        self.n_samples = n_samples
        self.resample_each = resample_each

        self.x = nnx.Variable(jnp.zeros((n_samples, 2)))
        self.counter = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    @abstractmethod
    def _sample(self, rngs: nnx.Rngs) -> jax.Array:
        raise NotImplementedError

    def __call__(self, rngs: nnx.Rngs) -> jax.Array:
        do_resample = self.counter.value == 0

        def resample_fn(rngs):
            return self._sample(rngs=rngs)

        def reuse_fn(*args):
            return self.x.value

        x = jax.lax.cond(
            do_resample,
            resample_fn,
            reuse_fn,
            operand=rngs,
        )

        self.x.value = x
        self.counter.value = (self.counter.value + 1) % self.resample_each

        return x


class CheckerboardBatcher(BaseBatcher):
    def __init__(self, n_samples: int, resample_each: int):
        super().__init__(n_samples=n_samples, resample_each=resample_each)

    def _sample(self, rngs: nnx.Rngs) -> jax.Array:
        points = rngs.data.uniform(
            shape=(self.n_samples, 2),
        )

        shifts_x = (
            rngs.data.randint(
                shape=(self.n_samples,),
                minval=0,
                maxval=4,
            )
            - 2
        )

        shifts_y = (
            rngs.data.randint(
                shape=(self.n_samples,),
                minval=0,
                maxval=2,
            )
            * 2
            + shifts_x % 2
            - 2
        )

        points = points.at[:, 0].add(shifts_x)
        points = points.at[:, 1].add(shifts_y)

        return points


class TwoSpiralsBatcher(BaseBatcher):
    def __init__(self, n_samples: int, resample_each: int):
        super().__init__(n_samples=n_samples, resample_each=resample_each)

    def _sample(self, rngs: nnx.Rngs) -> jax.Array:
        half = self.n_samples // 2

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

        d1y = jnp.sin(n) * n + rngs.data.uniform(shape=(n_samples - half, 1)) * 0.5

        x = (
            jnp.vstack(
                (
                    jnp.hstack((d1x, d1y)),
                    jnp.hstack((-d1x, -d1y)),
                )
            )
            / 3.0
        )

        x = x + rngs.data.uniform(shape=x.shape) * 0.1

        return x


class EightGaussiansBatcher(BaseBatcher):
    def __init__(self, n_samples: int, resample_each: int):
        super().__init__(n_samples=n_samples, resample_each=resample_each)

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
                shape=(self.n_samples, 2),
            )
            * 0.5
        )

        shift_ids = rngs.data.randint(
            shape=(self.n_samples,),
            minval=0,
            maxval=8,
        )

        x = blob + self.centers.value[shift_ids, :]
        x = x / 1.414

        return x


class LatentBatcherFromModel(BaseBatcher):
    def __init__(
        self, n_samples: int, resample_each: int, model: ParametricPushforward
    ):
        self._model = model
        super().__init__(n_samples=n_samples, resample_each=resample_each)

    def _sample(self, rngs):
        return self._model._sample_latent(self.n_samples, rngs)


def ZipBatcher(BaseBatcher):
    def __init__(
        self, n_samples: int, resample_each: int, *batchers: Tuple[BaseBatcher]
    ):
        super().__init__(n_samples=n_samples, resample_each=resample_each)
        self.batchers = batchers

    def _sample(self, rngs: nnx.Rngs):
        return tuple(b._sample(rngs) for b in batchers)
