import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "examples" / "flow_matching_image_benchmark_runner.py"


def load_runner():
    specification = importlib.util.spec_from_file_location("image_benchmark_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def smoke_config(directory):
    return {
        "experiment": {"name": "smoke", "seed": 1, "output_root": str(Path(directory) / "runs")},
        "problem": {
            "dataset": {"name": "mnist", "resolution": 28, "cache_dir": str(Path(directory) / "hf")},
            "encoder": {"type": "none"},
        },
        "rhs": {"type": "mlp", "flatten": True, "hidden_dims": [8]},
        "training": {"max_steps": 2, "batch_size": 4},
        "methods": [{"name": "adamw", "n_restarts": 1, "kwargs": {"learning_rate": 1e-3}}],
        "evaluation": {
            "val_fm_loss": {"enabled": True, "num_samples": 4, "batch_size": 2},
            "sampling": {"method": "euler", "steps": 1, "batch_size": 2},
            "fid": {"enabled": False},
            "kid": {"enabled": False}
        },
        "resources": {"gpu_ids": [], "gpus_per_run": 1, "max_concurrent_runs": 1, "worker_env": {}}
    }


def test_runner_import_and_validation_do_not_import_jax():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        payload = smoke_config(directory)
        path.write_text(json.dumps(payload), encoding="utf-8")
        script = f'''import importlib.util, sys
assert "jax" not in sys.modules
spec = importlib.util.spec_from_file_location("runner", {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "jax" not in sys.modules
assert module.main(["--config", {str(path)!r}, "--validate-only"]) == 0
assert "jax" not in sys.modules
'''
        subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def test_plot_only_does_not_prepare_dataset():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(smoke_config(directory)), encoding="utf-8")
        session = Path(directory) / "session"
        session.mkdir()
        called = []
        runner.prepare_dataset = lambda config: (_ for _ in ()).throw(
            AssertionError("plot-only prepared a dataset")
        )
        runner.plot_session = lambda value: called.append(Path(value))
        assert runner.main(
            ["--config", str(path), "--plot-only", "--output-dir", str(session)]
        ) == 0
        assert called == [session]


def test_worker_sets_environment_before_dispatch():
    runner = load_runner()

    class Queue:
        def __init__(self, values=None):
            self.values = list(values or [])

        def get(self):
            return self.values.pop(0)

        def put(self, value):
            self.values.append(value)

    config = smoke_config("/tmp")
    config["resources"]["worker_env"] = {"NEURIPP_TEST_WORKER": "set"}
    run = {"run_id": "run_test"}
    observed = {}
    original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    original_test = os.environ.get("NEURIPP_TEST_WORKER")
    try:
        def fake_run(*args):
            observed["cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            observed["worker"] = os.environ.get("NEURIPP_TEST_WORKER")
            return {"run_id": "run_test", "status": "completed"}

        runner._run_one = fake_run
        results = Queue()
        runner._worker_loop(config, "manifest", "/tmp", [2, 3], Queue([run, None]), results, False)
        assert observed == {"cuda": "2,3", "worker": "set"}
        assert results.values[0]["status"] == "completed"
    finally:
        if original_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
        if original_test is None:
            os.environ.pop("NEURIPP_TEST_WORKER", None)
        else:
            os.environ["NEURIPP_TEST_WORKER"] = original_test


def test_plot_only_rejects_run_selection():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(smoke_config(directory)), encoding="utf-8")
        try:
            runner.main(
                [
                    "--config",
                    str(path),
                    "--plot-only",
                    "--run-id",
                    "0",
                    "--output-dir",
                    str(Path(directory) / "session"),
                ]
            )
        except ValueError as error:
            assert "mutually exclusive" in str(error)
        else:
            raise AssertionError("Expected plot/run selection conflict")


if __name__ == "__main__":
    test_runner_import_and_validation_do_not_import_jax()
    test_plot_only_does_not_prepare_dataset()
    test_worker_sets_environment_before_dispatch()
    test_plot_only_rejects_run_selection()
    print("Image runner tests passed.")
