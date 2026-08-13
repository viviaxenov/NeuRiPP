from typing import Tuple

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
    X = X.reshape(N, -1)
    squared_norms = (X**2).sum(axis=-1)
    pairwise_sq_dists = jnp.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * X @ X.T,
        0.0,
    )
    idx = jnp.triu_indices(N, k=1)
    H = jnp.median(pairwise_sq_dists[idx])
    h = jnp.sqrt(0.5 * H / jnp.log(d + 1))

    return h


def gaussian_kernel(x1, x2, bw):
    return jnp.exp(-0.5 * ((x1 - x2) ** 2).sum() / bw)


gk = jnp.vectorize(gaussian_kernel, signature="(k),(k),()->()")


def gaussian_mmd(X1: jnp.ndarray, X2: jnp.ndarray, bandwidths: jnp.ndarray):
    X1 = X1.reshape(X1.shape[0], -1)
    X2 = X2.reshape(X2.shape[0], -1)

    def squared_distances(left, right):
        left_norms = (left**2).sum(axis=-1)
        right_norms = (right**2).sum(axis=-1)
        return jnp.maximum(
            left_norms[:, None] + right_norms[None, :] - 2.0 * left @ right.T,
            0.0,
        )

    def kernel_matrix(left, right):
        distances = squared_distances(left, right)
        return jnp.exp(
            -0.5 * distances[:, :, None] / bandwidths[None, None, :]
        ).sum(axis=-1)

    k_x1x1 = kernel_matrix(X1, X1)
    k_x1x2 = kernel_matrix(X1, X2)
    k_x2x2 = kernel_matrix(X2, X2)

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
