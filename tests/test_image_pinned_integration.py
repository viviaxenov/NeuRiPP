"""Explicit installed DiffuseNNX smoke test; run as a script on an accelerator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from image_benchmarks.assets.diffuse_nnx import (
    DIFFUSE_NNX_COMMIT,
    require_diffuse_nnx,
)
from image_benchmarks.assets.files import (
    INCEPTION_WEIGHTS_SHA256,
    prepare_inception_weights,
)
from image_benchmarks.evaluation.inception import DiffuseInceptionFeatures
from image_benchmarks.rhs.registry import build_rhs, parameter_count


EXPECTED_SIT_PARAMETERS = {"S": 32_474_640, "B": 129_534_864}


def run_pinned_integration(weights_path: str | Path, *, auto_download: bool) -> None:
    version = require_diffuse_nnx()
    metadata = prepare_inception_weights(
        weights_path,
        auto_download=auto_download,
        expected_sha256=INCEPTION_WEIGHTS_SHA256,
    )
    extractor = DiffuseInceptionFeatures(
        metadata["path"], expected_sha256=INCEPTION_WEIGHTS_SHA256
    )
    features = extractor(np.zeros((1, 32, 32, 3), dtype=np.uint8))
    assert features.shape == (1, 2048)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()

    state_shape = (32, 32, 4)
    for seed, variant in enumerate(("S", "B"), start=1):
        rhs = build_rhs(
            {
                "type": "sit",
                "implementation": "diffuse_nnx",
                "variant": variant,
                "patch_size": 2,
                "class_conditioning": False,
            },
            state_shape,
            rngs=nnx.Rngs(seed),
        )
        output = rhs(0.5, jnp.zeros(state_shape, dtype=jnp.float32))
        jax.block_until_ready(output)
        assert output.shape == state_shape
        assert bool(jnp.isfinite(output).all())
        assert parameter_count(rhs) == EXPECTED_SIT_PARAMETERS[variant]
    print(f"DiffuseNNX {version} ({DIFFUSE_NNX_COMMIT}) integration passed.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--no-auto-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pinned_integration(
        arguments.weights_path, auto_download=not arguments.no_auto_download
    )
