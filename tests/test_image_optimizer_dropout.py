"""Short JIT smoke test for optimizer behavior with U-Net dropout."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from flax import nnx

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from image_benchmarks.rhs.registry import build_rhs
from image_benchmarks.training.trainer import ImageTrainer
from neuripp.parametric_pushforward.flow_matching import FlowMatching, flow_matching_loss


def _tiny_flow_matching(seed: int):
    rngs = nnx.Rngs(seed)
    rhs = build_rhs(
        {
            "type": "unet",
            "variant": "small",
            "base_channels": 4,
            "channel_mult": [1],
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "dropout": 0.1,
        },
        (4, 4, 1),
        rngs=rngs,
    )
    return FlowMatching(rhs, rngs, 2, ode_method="euler", ode_nstep_max=1)


def _run_ten_steps(method_config, seed):
    model = _tiny_flow_matching(seed)
    batch = jax.random.normal(jax.random.key(seed + 1), (2, 4, 4, 1))
    trainer = ImageTrainer(
        model,
        method_config,
        flow_matching_loss,
        batch,
        nnx.Rngs(seed + 2),
    )
    for _ in range(10):
        values = trainer.step(batch)
        assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(values))
    assert trainer.step_count == 10
    assert model.rhs.down_stages[0].blocks[0].dropout.deterministic is False


def test_unet_dropout_runs_for_adam_ngd_and_anderson():
    methods = [
        {"name": "adam", "kwargs": {"learning_rate": 1e-3}},
        {
            "name": "ngd",
            "kwargs": {
                "step_size": 1e-3,
                "linear_solver_regularization": 1e-3,
                "linear_solver_tolerance": 1e-6,
                "linear_solver_maxiter": 3,
                "regularization_factor": 1e-3,
            },
        },
        {
            "name": "anderson",
            "kwargs": {
                "step_size": 1e-3,
                "linear_solver_regularization": 1e-3,
                "linear_solver_tolerance": 1e-6,
                "linear_solver_maxiter": 3,
            },
        },
    ]
    for seed, method in enumerate(methods, start=10):
        _run_ten_steps(method, seed)


if __name__ == "__main__":
    test_unet_dropout_runs_for_adam_ngd_and_anderson()
    print("Image optimizer dropout test passed.")
