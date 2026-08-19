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
    sys.path.insert(0, str(ROOT / "examples"))
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


def test_plot_session_overlays_training_loss_by_step_and_time():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        session = Path(directory)
        (session / "plots").mkdir()
        runs = [
            {"run_id": "adam", "method": {"name": "adamw"}, "restart_index": 0},
            {"run_id": "ngd", "method": {"name": "ngd"}, "restart_index": 0},
        ]
        (session / "planned_runs.json").write_text(json.dumps(runs), encoding="utf-8")
        for index, run in enumerate(runs):
            run_dir = session / "runs" / run["run_id"]
            run_dir.mkdir(parents=True)
            records = [
                {
                    "type": "train",
                    "optimizer_step": step,
                    "wall_clock_train_s": step * (index + 1),
                    "loss": 10.0 / step,
                }
                for step in (1, 2)
            ]
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
        runner.plot_session(session)
        for filename in (
            "diagnostics_comparison.pdf",
        ):
            assert (session / "plots" / filename).stat().st_size > 0


def test_save_run_diagnostics_produces_pdf():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "runs" / "run_0000"
        run_dir.mkdir(parents=True)
        records = [
            {
                "type": "train",
                "optimizer_step": step,
                "wall_clock_train_s": step,
                "loss": 10.0 / step,
            }
            for step in (1, 2, 3)
        ] + [
            {
                "type": "validation",
                "optimizer_step": step,
                "wall_clock_train_s": step,
                "val_fm_loss": 5.0 / step,
            }
            for step in (2, 3)
        ] + [
            {
                "type": "validation_ema",
                "optimizer_step": step,
                "wall_clock_train_s": step,
                "val_fm_loss": 4.0 / step,
            }
            for step in (2, 3)
        ] + [
            {
                "type": "validation_sw",
                "optimizer_step": step,
                "wall_clock_train_s": step,
                "sliced_wasserstein": 1.0 / step,
            }
            for step in (2, 3)
        ]
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        run = {"run_id": "run_0000", "run_index": 0, "method": {"name": "adamw", "kwargs": {}}}
        diagnostic_dir = Path(directory) / "plots" / "diagnostic"
        runner._save_run_diagnostics(diagnostic_dir, run_dir, run)
        assert (diagnostic_dir / "diagnostics_run_0000.pdf").stat().st_size > 0


def test_run_label_uses_shorthand_and_only_varied_keys():
    runner = load_runner()
    runs = [
        {
            "method": {
                "name": "ngd",
                "kwargs": {
                    "step_size": 0.001,
                    "linear_solver_regularization": 0.01,
                    "natural_grad_clipping_threshold": 20.0,
                    "linear_solver_tolerance": 1e-6,
                    "linear_solver_maxiter": 50,
                },
            }
        },
        {
            "method": {
                "name": "ngd",
                "kwargs": {
                    "step_size": 0.01,
                    "linear_solver_regularization": 0.001,
                    "natural_grad_clipping_threshold": None,
                    "linear_solver_tolerance": 1e-6,
                    "linear_solver_maxiter": 50,
                },
            }
        },
    ]
    varied = runner._varying_run_keys(runs)
    assert "step_size" in varied
    assert "natural_grad_clipping_threshold" in varied
    assert "linear_solver_tolerance" not in varied
    label = runner._run_label(runs[0], varied)
    assert label.startswith("NGD")
    assert r"$h$=0.001" in label
    assert r"$\Lambda$=0.01" in label
    assert r"$\|\mathrm{grad}E\|_{\max}$=20" in label


def test_apply_log_ylim_anchors_to_step0_max():
    runner = load_runner()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    curves = [
        ([0, 1, 2], [10.0, 1.0, 0.1]),
        ([0, 1, 2], [5.0, 2.0, 1000.0]),  # diverges late; must not set the top
    ]
    runner._apply_log_ylim(axis, curves)
    _, top = axis.get_ylim()
    assert top == 10.0
    assert axis.get_yscale() == "log"
    plt.close(figure)


def test_best_validation_loss_prefers_final_summary():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        session = Path(directory)
        run = {"run_id": "r0", "method": {"name": "ngd"}}
        run_dir = session / "runs" / "r0"
        run_dir.mkdir(parents=True)
        (run_dir / "final_summary.json").write_text(
            json.dumps({"best_val_fm_loss": 123.4}), encoding="utf-8"
        )
        # metrics.jsonl with a worse validation loss; final_summary should win
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"type": "validation", "val_fm_loss": 999.0}) + "\n",
            encoding="utf-8",
        )
        assert runner._best_validation_loss(session, run) == 123.4


def test_best_validation_loss_falls_back_to_min_validation_record():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        session = Path(directory)
        run = {"run_id": "r0", "method": {"name": "ngd"}}
        run_dir = session / "runs" / "r0"
        run_dir.mkdir(parents=True)
        records = [
            {"type": "validation", "val_fm_loss": 50.0},
            {"type": "validation", "val_fm_loss": 30.0},
        ]
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        assert runner._best_validation_loss(session, run) == 30.0


def test_select_topk_runs_per_method_by_best_validation():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        session = Path(directory)
        specs = [("ngd", 5.0), ("ngd", 3.0), ("ngd", 1.0), ("adamw", 4.0), ("adamw", 2.0)]
        runs_data = []
        for i, (method, bv) in enumerate(specs):
            rid = f"r{i}"
            run = {"run_id": rid, "method": {"name": method}}
            run_dir = session / "runs" / rid
            run_dir.mkdir(parents=True)
            (run_dir / "final_summary.json").write_text(
                json.dumps({"best_val_fm_loss": bv}), encoding="utf-8"
            )
            runs_data.append((run, [], []))
        # k=2: ngd keeps 1.0 (idx 2) and 3.0 (idx 1); adamw keeps 2.0 (idx 4) and 4.0 (idx 3)
        selected, order = runner._select_topk_runs(session, runs_data, k=2)
        assert order == [1, 2, 3, 4]
        assert [run["run_id"] for run, _, _ in selected] == ["r1", "r2", "r3", "r4"]
        # k=1: only the best per method
        _, order1 = runner._select_topk_runs(session, runs_data, k=1)
        assert order1 == [2, 4]


def test_plot_session_produces_topk_pdf():
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        session = Path(directory)
        (session / "plots").mkdir(parents=True)
        runs = [
            {"run_id": "a1", "method": {"name": "adamw", "kwargs": {}}, "restart_index": 0},
            {"run_id": "n1", "method": {"name": "ngd", "kwargs": {}}, "restart_index": 0},
        ]
        (session / "planned_runs.json").write_text(json.dumps(runs), encoding="utf-8")
        (session / "resolved_config.json").write_text(
            json.dumps({"plotting": {"top_runs_per_method": 1}}), encoding="utf-8"
        )
        for run in runs:
            run_dir = session / "runs" / run["run_id"]
            run_dir.mkdir(parents=True)
            records = [
                {"type": "train", "optimizer_step": 1, "wall_clock_train_s": 1, "loss": 10.0}
            ]
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
            )
            (run_dir / "final_summary.json").write_text(
                json.dumps({"best_val_fm_loss": 1.0}), encoding="utf-8"
            )
        runner.plot_session(session)
        assert (session / "plots" / "diagnostics_comparison.pdf").stat().st_size > 0
        assert (session / "plots" / "diagnostics_top1.pdf").stat().st_size > 0


if __name__ == "__main__":
    test_runner_import_and_validation_do_not_import_jax()
    test_plot_only_does_not_prepare_dataset()
    test_worker_sets_environment_before_dispatch()
    test_plot_only_rejects_run_selection()
    test_plot_session_overlays_training_loss_by_step_and_time()
    test_save_run_diagnostics_produces_pdf()
    test_run_label_uses_shorthand_and_only_varied_keys()
    test_apply_log_ylim_anchors_to_step0_max()
    test_best_validation_loss_prefers_final_summary()
    test_best_validation_loss_falls_back_to_min_validation_record()
    test_select_topk_runs_per_method_by_best_validation()
    test_plot_session_produces_topk_pdf()
    print("Image runner tests passed.")
