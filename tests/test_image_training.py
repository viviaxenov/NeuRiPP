import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from neuripp.image_benchmarks.distributed import DataParallelContext
from neuripp.image_benchmarks.training.methods import resolve_method
from neuripp.image_benchmarks.training.trainer import ImageTrainer
from neuripp.parametric_pushforward.flow_matching import FlowMatching, flow_matching_loss


class TinyRHS(nnx.Module):
    def __init__(self, rngs):
        self.dim = 2
        self.linear = nnx.Linear(3, 2, rngs=rngs)

    def __call__(self, time, state, *args):
        del args
        return self.linear(jnp.concatenate((state, jnp.atleast_1d(time))))


def make_model(seed):
    rngs = nnx.Rngs(seed)
    return FlowMatching(TinyRHS(rngs), rngs, 4, ode_method="euler", ode_nstep_max=1)


METHOD_CONFIGS = {
    "adamw": {
        "name": "adamw",
        "kwargs": {
            "learning_rate": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
    },
    "ngd": {
        "name": "ngd",
        "kwargs": {
            "step_size": 1e-3,
            "linear_solver_regularization": 1e-2,
            "linear_solver_maxiter": 2,
        },
    },
    "anderson": {
        "name": "anderson",
        "kwargs": {
            "step_size": 1e-3,
            "regularization_factor": 1e-2,
            "history_length": 2,
            "linear_solver_regularization": 1e-2,
            "linear_solver_maxiter": 2,
        },
    },
}


def _worker(device_count):
    context = DataParallelContext.create(expected_device_count=device_count)
    data = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    results = {}
    for method_index, (name, config) in enumerate(METHOD_CONFIGS.items()):
        trainer = ImageTrainer(
            make_model(100 + method_index),
            config,
            flow_matching_loss,
            data,
            nnx.Rngs(200 + method_index),
            data_parallel=context,
        )
        values = None
        for _ in range(2):
            values = trainer.step(data)
        parameters = nnx.state(trainer.model, nnx.Param)
        flat = np.concatenate(
            [np.asarray(value).reshape(-1) for value in jax.tree.leaves(parameters)]
        )
        results[name] = {
            "metrics": [float(np.asarray(value)) for value in values],
            "parameters": flat.tolist(),
            "accounting": trainer.accounting(),
        }
    print("PARITY_JSON=" + json.dumps(results, sort_keys=True))


def _run_parity_worker(device_count):
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    output = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--parity-worker", str(device_count)],
        env=environment,
        text=True,
        timeout=300,
    )
    line = next(value for value in output.splitlines() if value.startswith("PARITY_JSON="))
    return json.loads(line.split("=", 1)[1])


def test_method_registry_resolves_existing_factories():
    for config in METHOD_CONFIGS.values():
        resolved = resolve_method(config, flow_matching_loss)
        assert callable(resolved.init_fn)
        assert callable(resolved.step_fn)
    assert resolve_method(METHOD_CONFIGS["anderson"], flow_matching_loss).initialization_updates == 1


def test_data_parallel_rejects_indivisible_global_batch():
    context = DataParallelContext.create()
    invalid_count = context.device_count + 1 if context.device_count > 1 else 0
    if invalid_count:
        try:
            context.shard_batch(np.zeros((invalid_count, 2), dtype=np.float32))
        except ValueError as error:
            assert "not divisible" in str(error)
        else:
            raise AssertionError("Expected indivisible batch rejection")


def test_one_and_two_device_updates_match():
    one = _run_parity_worker(1)
    two = _run_parity_worker(2)
    for method in METHOD_CONFIGS:
        np.testing.assert_allclose(
            one[method]["metrics"], two[method]["metrics"], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            one[method]["parameters"],
            two[method]["parameters"],
            rtol=1e-5,
            atol=1e-6,
        )
        assert one[method]["accounting"]["optimizer_step"] == two[method]["accounting"]["optimizer_step"]
        assert one[method]["accounting"]["examples_seen"] == two[method]["accounting"]["examples_seen"]


def test_trainer_checkpoint_payload_restores_anderson_exactly():
    data = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    original = ImageTrainer(
        make_model(40),
        METHOD_CONFIGS["anderson"],
        flow_matching_loss,
        data,
        nnx.Rngs(41),
        dataset_size=8,
    )
    original.step(data)
    payload = original.checkpoint_payload()
    restored = ImageTrainer(
        make_model(40),
        METHOD_CONFIGS["anderson"],
        flow_matching_loss,
        data,
        nnx.Rngs(41),
        dataset_size=8,
    )
    restored.restore_checkpoint_payload(payload)
    original_values = original.step(data)
    restored_values = restored.step(data)
    np.testing.assert_allclose(
        [float(value) for value in original_values],
        [float(value) for value in restored_values],
        rtol=1e-6,
    )
    original_parameters = jax.tree.leaves(nnx.state(original.model, nnx.Param))
    restored_parameters = jax.tree.leaves(nnx.state(restored.model, nnx.Param))
    for left, right in zip(original_parameters, restored_parameters, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6)
    assert restored.accounting()["effective_epoch"] == original.accounting()["effective_epoch"]


if __name__ == "__main__":
    if "--parity-worker" in sys.argv:
        index = sys.argv.index("--parity-worker")
        _worker(int(sys.argv[index + 1]))
    else:
        test_method_registry_resolves_existing_factories()
        test_data_parallel_rejects_indivisible_global_batch()
        test_one_and_two_device_updates_match()
        test_trainer_checkpoint_payload_restores_anderson_exactly()
        print("Image training tests passed.")
