"""Kernel Inception Distance over cached Inception features."""

from __future__ import annotations

import numpy as np


def polynomial_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("KID features must be matrices with equal feature dimension")
    return (left @ right.T / left.shape[1] + 1.0) ** 3


def polynomial_mmd_unbiased(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        raise ValueError("KID subsets require at least two examples")
    kernel_left = polynomial_kernel(left, left)
    kernel_right = polynomial_kernel(right, right)
    kernel_cross = polynomial_kernel(left, right)
    left_term = (kernel_left.sum() - np.trace(kernel_left)) / (
        len(left) * (len(left) - 1)
    )
    right_term = (kernel_right.sum() - np.trace(kernel_right)) / (
        len(right) * (len(right) - 1)
    )
    return float(left_term + right_term - 2.0 * kernel_cross.mean())


def calculate_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    *,
    subsets: int = 100,
    subset_size: int = 1000,
    seed: int = 0,
) -> dict[str, float | int]:
    real_features = np.asarray(real_features)
    fake_features = np.asarray(fake_features)
    if subsets < 1:
        raise ValueError("KID subsets must be positive")
    size = min(subset_size, len(real_features), len(fake_features))
    if size < 2:
        raise ValueError("Not enough features for KID")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(subsets):
        real_indices = rng.choice(len(real_features), size=size, replace=False)
        fake_indices = rng.choice(len(fake_features), size=size, replace=False)
        values.append(
            polynomial_mmd_unbiased(
                real_features[real_indices], fake_features[fake_indices]
            )
        )
    values_array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(values_array.std(ddof=1)) if subsets > 1 else 0.0
    return {
        "kid_mean": float(values_array.mean()),
        "kid_std": standard_deviation,
        "kid_stderr": standard_deviation / np.sqrt(subsets),
        "kid_subsets": int(subsets),
        "kid_subset_size": int(size),
    }
