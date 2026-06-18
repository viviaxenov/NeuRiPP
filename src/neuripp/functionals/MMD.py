from typing import Callable

import jax
import jax.numpy as jnp

from ..parametric_pushforward.parametric_pushforward import ParametricPushforward


def bandwidth_median(X: jnp.array) -> float:
    """Estimate a suitable bandwidth for the kernel using the median heuristic.

    Args:
        X: array of shape `(N_samples, dim)`, representing the sample

    Returns:
        float: the bandwidth
    """
    N, d = X.shape
    X_diffs = X[jnp.newaxis, :, :] - X[:, jnp.newaxis, :]
    idx = jnp.triu_indices(N, k=1)
    X_diffs = X_diffs[*idx, :]
    pairwise_sq_dists = (X_diffs**2).sum(axis=-1)
    H = jnp.median(pairwise_sq_dists)
    h = jnp.sqrt(0.5 * H / jnp.log(d + 1))

    return h


def gaussian_kernel(x1, x2, bw):
    return jnp.exp(-0.5 * ((x1 - x2) ** 2).sum() / bw)


gk = jnp.vectorize(gaussian_kernel, signature="(k),(k),()->()")


def gaussian_mmd(X1: jnp.ndarray, X2: jnp.ndarray, bandwidths: jnp.ndarray):
    X1 = X1.reshape(X1.shape[0], -1)
    X2 = X2.reshape(X2.shape[0], -1)

    k_x1x1 = gk(
        X1[:, None, None, :], X1[None, :, None, :], bandwidths[None, None, :]
    ).sum(axis=-1)
    k_x1x2 = gk(
        X1[:, None, None, :], X2[None, :, None, :], bandwidths[None, None, :]
    ).sum(axis=-1)
    k_x2x2 = gk(
        X2[:, None, None, :], X2[None, :, None, :], bandwidths[None, None, :]
    ).sum(axis=-1)

    d1 = X1.shape[0]
    d2 = X2.shape[0]
    A = jnp.triu(k_x1x1, k=1).sum() / (d1 * (d1 - 1))
    C = jnp.triu(k_x2x2, k=1).sum() / (d2 * (d2 - 1))
    B = k_x1x2.mean()

    mmd_sq = 2.0 * (A - B + C)

    return jnp.maximum(0.0, mmd_sq) ** 0.5


def getMMD(
    batch_size: int, data: jnp.ndarray = None, bw_multipliers: jnp.ndarray = None
):
    bw_base = 1.0
    if data is not None:
        bw_base = bandwidth_median(data)
    if bw_multipliers is None:
        bw_multipliers = jnp.array([1.0])

    bandwidths = bw_multipliers * bw_base

    def _MMD(model: ParametricPushforward, data_batch: jnp.ndarray):
        x = model.sample(batch_size)
        return gaussian_mmd(x, data_batch, bandwidths)

    return _MMD


def checkerboard_generator(n_samples: int, resample_each: int, seed: int = 3):
    key = jax.random.PRNGKey(seed)
    while True:
        key, points_key, shift_x_key, shift_y_key = jax.random.split(key, 4)
        points = jax.random.uniform(points_key, (n_samples, 2))
        shifts_x = jax.random.randint(shift_x_key, (n_samples,), 0, 4) - 2
        shifts_y = (
            jax.random.randint(shift_y_key, (n_samples,), 0, 2) * 2 + shifts_x % 2 - 2
        )
        points = points.at[:, 0].add(shifts_x)
        points = points.at[:, 1].add(shifts_y)
        for _ in range(resample_each):
            yield points


def two_spirals_generator(n_samples: int, resample_each: int, seed: int = 3):
    key = jax.random.PRNGKey(seed)
    while True:
        key, n_key, shift_x_key, shift_y_key, noise_key = jax.random.split(key, 5)
        n = (
            jnp.sqrt(jax.random.uniform(n_key, (n_samples // 2, 1)))
            * 540
            * (2 * jnp.pi)
            / 360
        )
        d1x = (
            -jnp.cos(n) * n + jax.random.uniform(shift_x_key, (n_samples // 2, 1)) * 0.5
        )
        d1y = (
            jnp.sin(n) * n + jax.random.uniform(shift_y_key, (n_samples // 2, 1)) * 0.5
        )
        x = jnp.vstack((jnp.hstack((d1x, d1y)), jnp.hstack((-d1x, -d1y)))) / 3
        x += jax.random.uniform(noise_key, x.shape) * 0.1
        for _ in range(resample_each):
            yield x


def eight_gaussians_generator(n_samples: int, resample_each: int, seed: int = 3):
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 8)
    centers = 4.0 * jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1)
    key = jax.random.PRNGKey(seed)

    while True:
        key, idx_key, blob_key = jax.random.split(key, 3)
        blob = jax.random.normal(blob_key, (n_samples, 2)) * 0.5
        shift_ids = jax.random.randint(idx_key, n_samples, minval=0, maxval=7)

        x = blob + centers[shift_ids, :]
        x /= 1.414

        for _ in range(resample_each):
            yield x


def file_dataset_generator(file_path: str, n_samples: int, resample_each: int):
    raise NotImplementedError(
        "File-based dataset generator is not implemented yet "
        f"(requested file: {file_path})"
    )


