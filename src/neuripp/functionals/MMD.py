from typing import Callable, Tuple

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
    data: jnp.ndarray = None, bw_multipliers: jnp.ndarray = None, bw_base: float = None,
):
    if data is not None:
        bw_base = bandwidth_median(data)
    elif bw_base is None:
        raise ValueError("Failed to determine bandwidth! Either give bw_base directly or a data batch to use with the median heuristic")

    if bw_multipliers is None:
        bw_multipliers = jnp.array([1.0])

    bandwidths = bw_multipliers * bw_base

    def _MMD(model: ParametricPushforward, batch: Tuple[jnp.ndarray]):
        latent_batch, data_batch = batch
        x = model(latent_batch)
        return gaussian_mmd(x, data_batch, bandwidths)

    return _MMD
