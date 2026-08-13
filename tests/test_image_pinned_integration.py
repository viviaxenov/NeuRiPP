"""Explicit pinned DiffuseNNX smoke test; run as a script on an accelerator."""

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

from image_benchmarks.assets.diffuse_nnx import prepare_diffuse_nnx_source
from image_benchmarks.evaluation.inception import DiffuseInceptionFeatures
from image_benchmarks.rhs.registry import build_rhs, parameter_count


EXPECTED_SIT_PARAMETERS = {"S": 32_474_640, "B": 129_534_864}


def run_pinned_integration(source_dir: str | Path) -> None:
    source = prepare_diffuse_nnx_source(source_dir, auto_download=True)
    extractor = DiffuseInceptionFeatures(source)
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
                "source_dir": str(source),
                "source_auto_download": False,
            },
            state_shape,
            rngs=nnx.Rngs(seed),
        )
        output = rhs(0.5, jnp.zeros(state_shape, dtype=jnp.float32))
        jax.block_until_ready(output)
        assert output.shape == state_shape
        assert bool(jnp.isfinite(output).all())
        assert parameter_count(rhs) == EXPECTED_SIT_PARAMETERS[variant]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pinned_integration(arguments.source_dir)
    print("Pinned DiffuseNNX SiT/Inception integration passed.")
