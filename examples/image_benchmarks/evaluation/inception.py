"""Thin adapter around the installed DiffuseNNX FID Inception implementation."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from image_benchmarks.assets.diffuse_nnx import import_diffuse_module
from image_benchmarks.assets.files import INCEPTION_WEIGHTS_SHA256


class DiffuseInceptionFeatures:
    """Single-process feature extractor backed by verified external weights."""

    feature_dim = 2048
    provenance = "diffuse_nnx_inception_v3_fid_da5f2b7_4e030efa"

    def __init__(
        self,
        weights_path: str | Path,
        *,
        expected_sha256: str = INCEPTION_WEIGHTS_SHA256,
    ):
        module = import_diffuse_module("diffuse_nnx.eval.inception")
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.expected_sha256 = expected_sha256
        self.detector = module.InceptionV3(
            pretrained=True,
            weights_path=str(self.weights_path),
            expected_sha256=expected_sha256,
        )
        self.variables = self.detector.init(
            jax.random.key(0), jnp.ones((1, 299, 299, 3), dtype=jnp.float32)
        )

        def apply_detector(images):
            images = images.astype(jnp.float32) / 127.5 - 1.0
            images = jax.image.resize(
                images, (images.shape[0], 299, 299, 3), method="bilinear"
            )
            return self.detector.apply(self.variables, images, train=False).reshape(
                images.shape[0], -1
            )

        self._apply = jax.jit(apply_detector)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim != 4 or images.dtype != np.uint8:
            raise ValueError("Inception images must be uint8 NHWC batches")
        if images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        if images.shape[-1] != 3:
            raise ValueError("Inception images must have one or three channels")
        return np.asarray(self._apply(jnp.asarray(images)))
