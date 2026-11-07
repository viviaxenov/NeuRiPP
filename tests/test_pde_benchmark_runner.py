"""Focused runner tests: no PDE solver or JAX import is needed."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import benchmark_runner as runner


class FieldRunnerTests(unittest.TestCase):
    def config(self, name):
        return runner.load_config(ROOT / "examples" / "configs" / f"kl_{name}.json")

    def test_presets_and_smoke_budgets(self):
        for name, dim in (("acoustic", 65536), ("acoustic_128", 16384),
                          ("acoustic_64", 4096), ("ks", 200), ("ks_64", 64),
                          ("acoustic_smoke", 4096), ("ks_smoke", 64)):
            with self.subTest(name=name):
                config = self.config(name)
                runs = runner.plan_runs(config)
                self.assertEqual([r["method"] for r in runs], ["adam", "ngd"])
                self.assertTrue(all(r["resolved_problem"]["dim"] == dim for r in runs))
                self.assertEqual(config["common_params"]["batch_size"], 1)
                if "smoke" in name:
                    self.assertEqual(config["common_params"]["max_iterations"], 1)
                    self.assertEqual(config["common_params"]["N_monte_carlo"], 1)
                    self.assertEqual(runs[0]["architecture"]["ode_nstep_max"], 2)
                    self.assertEqual(runs[0]["architecture"]["ode_method"], "euler")
                    self.assertEqual(runs[1]["method_kwargs"]["linear_solver_maxiter"], 3)
        self.assertEqual(self.config("acoustic")["problem"]["distribution"]["source_batch_size"], 8)
        self.assertEqual(self.config("acoustic_64")["problem"]["distribution"]["source_batch_size"], 1)

    def test_defaults_validation_and_dimension(self):
        for name in ("acoustic", "ks"):
            config = self.config(name)
            problem = config["problem"]
            architecture = config["architectures"][0]
            for field, value in (("truth_seed", True), ("observation_seed", -1),
                                 ("resample_each", 0), ("typo", 1)):
                bad = copy.deepcopy(problem)
                bad["distribution"][field] = value
                with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                    runner.resolve_problem(bad, {}, architecture)
            for update in ({"dim": 2}, {"functional": {"kind": "MMD"}}):
                with self.assertRaises(ValueError):
                    runner.resolve_problem({**problem, **update}, {}, architecture)
            with self.assertRaises(ValueError):
                runner.resolve_problem(problem, {}, {"rhs": {"model": "MLP"}})
        for update in ({"checkpoint_strategy": "bad"}, {"num_sources": 0},
                       {"data_source_batch_size": False}, {"n_grid": 7},
                       {"n_grid": 8, "num_sources": 64}):
            with self.assertRaises(ValueError):
                runner.resolve_field_distribution({"name": "acoustic", **update})
        with self.assertRaises(ValueError):
            runner.resolve_field_distribution({"name": "ks", "num_points": 8, "num_sensors": 9})
        settings = runner.resolve_field_distribution({"name": "acoustic", "source_batch_size": 8})
        self.assertEqual(settings["data_source_batch_size"], 8)
        self.assertEqual(settings["observation_seed"], 43)

    def test_cache_all_constructor_settings(self):
        for name in ("acoustic", "ks"):
            runner._FIELD_LOGPDF_CACHE.clear()
            calls = []
            def getter(**kwargs):
                calls.append(kwargs)
                return object()
            deps = {f"get_logpdf_{name}_inverse_problem": getter}
            base = {"name": name}
            first = runner.build_field_logpdf(base, deps)
            self.assertIs(first, runner.build_field_logpdf(
                {"name": name, **runner.resolve_field_distribution(base)}, deps))
            for key, value in runner.resolve_field_distribution(base).items():
                changed = "solver" if key == "checkpoint_strategy" else value + 1
                self.assertIsNot(first, runner.build_field_logpdf({**base, key: changed}, deps))
            self.assertEqual(len(calls), 1 + len(runner.resolve_field_distribution(base)))

    def test_optional_import_error_and_rhs_dispatch(self):
        runner._FIELD_LOGPDF_CACHE.clear()
        def missing(**kwargs):
            raise ModuleNotFoundError("missing", name="exponax")
        with self.assertRaisesRegex(ModuleNotFoundError, "optional dependency 'exponax'"):
            runner.build_field_logpdf({"name": "ks"}, {"get_logpdf_ks_inverse_problem": missing})
        class Module:
            @staticmethod
            def KSDilatedResNetRHS(dim, **kwargs):
                return dim, kwargs
        with patch.object(runner, "import_module", return_value=Module) as load:
            result = runner.build_rhs({"rhs": {"model": "KSDilatedResNetRHS"}}, 64,
                                      rngs="rng", deps={"rhs_registry": {}})
        load.assert_called_once_with("rhs_ks")
        self.assertEqual(result, (64, {"rngs": "rng"}))

    def test_elliptic_regression(self):
        config = self.config("elliptic")
        self.assertEqual(config["resolved_problem"]["dim"], 20)
        self.assertEqual(len(runner.plan_runs(config)), 12)
        runner._ELLIPTIC_LOGPDF_CACHE.clear()
        calls = []
        def getter(**kwargs):
            calls.append(kwargs)
            return "logpdf"
        distribution = config["problem"]["distribution"]
        deps = {"get_logpdf_elliptic_inverse_problem": getter}
        self.assertEqual(runner.build_elliptic_logpdf(20, distribution, deps), "logpdf")
        runner.build_elliptic_logpdf(20, distribution, deps)
        self.assertEqual(calls, [dict(seed=1, dim=20, n_grid=2000, noise_std=0.001)])

    def test_planning_in_fresh_process_blocks_numerical_imports(self):
        code = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'jax', 'flax', 'jwave', 'exponax', 'logpdf_targets', 'rhs_acoustic', 'rhs_ks'}:
            raise AssertionError('Unexpected planning import: ' + fullname)
sys.meta_path.insert(0, Block())
import benchmark_runner as r
from pathlib import Path
for path in Path('examples/configs').glob('kl_*.json'):
    r.plan_runs(r.load_config(path))
'''
        import os
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "examples")},
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
