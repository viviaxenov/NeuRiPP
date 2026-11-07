"""Small, dependency-light tests for notebook posterior summaries and plots."""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import inverse_pde_visualization as visualization
import benchmark_runner as runner


class _FakeRngs:
    def __init__(self, seed):
        self.seed = seed


class _FakeNNX:
    Rngs = _FakeRngs


class _FakeJnp:
    concatenate = staticmethod(np.concatenate)


class _FakeModel:
    def __init__(self):
        self.calls = []

    def sample(self, n_samples, rngs, with_log_density=False):
        self.calls.append((n_samples, rngs.seed, with_log_density))
        start = sum(call[0] for call in self.calls[:-1])
        samples = np.arange(start, start + n_samples, dtype=float)[:, None]
        return samples, samples[:, 0] + 100.0


class PosteriorVisualizationTests(unittest.TestCase):
    def setUp(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_sample_model_posterior_batches_and_returns_model_log_density(self):
        model = _FakeModel()
        samples, log_density = visualization.sample_model_posterior(
            model,
            n_samples=5,
            seed=17,
            batch_size=2,
            deps={"nnx": _FakeNNX, "jnp": _FakeJnp},
        )
        np.testing.assert_array_equal(samples[:, 0], np.arange(5))
        np.testing.assert_array_equal(log_density, np.arange(5) + 100.0)
        self.assertEqual(model.calls, [(2, 17, True), (2, 17, True), (1, 17, True)])

    def test_summary_selects_neural_ode_density_map_and_pointwise_stats(self):
        samples = np.array([[0.0, 1.0], [1.0, 3.0], [5.0, 2.0], [9.0, 8.0]])
        log_density = np.array([-4.0, -1.0, -3.0, np.nan])
        summary = visualization.summarize_field_samples(samples, log_density)

        self.assertEqual(summary["map_index"], 1)
        np.testing.assert_array_equal(summary["map_sample"], samples[1])
        self.assertEqual(summary["map_log_density"], -1.0)
        np.testing.assert_allclose(summary["mean"], np.mean(samples, axis=0))
        np.testing.assert_allclose(summary["std"], np.std(samples, axis=0))
        np.testing.assert_allclose(summary["lower"], np.quantile(samples, 0.1, axis=0))
        np.testing.assert_allclose(summary["upper"], np.quantile(samples, 0.9, axis=0))

    def test_summary_rejects_bad_samples_and_log_density(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            visualization.summarize_field_samples(np.zeros(4), np.zeros(1))
        with self.assertRaisesRegex(ValueError, "at least one finite"):
            visualization.summarize_field_samples(np.zeros((2, 3)), np.array([np.nan, -np.inf]))

    def test_acoustic_plot_has_four_requested_panels(self):
        import matplotlib.pyplot as plt

        samples = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
        summary = visualization.summarize_field_samples(samples, np.array([0.0, 1.0]))
        fig, axes = visualization.plot_acoustic_summary(np.zeros(4), summary)
        self.assertEqual(len(axes), 4)
        self.assertEqual(
            [axis.get_title() for axis in axes],
            ["Ground truth", "Neural-ODE MAP sample", "Sample mean", "Sample std"],
        )
        plt.close(fig)

    def test_ks_plot_has_truth_mean_map_and_credible_band(self):
        import matplotlib.pyplot as plt

        samples = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        summary = visualization.summarize_field_samples(samples, np.array([0.0, 2.0, 1.0]))
        fig, axis = visualization.plot_ks_summary(np.zeros(3), summary)
        self.assertEqual(len(axis.lines), 3)
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(
            [line.get_label() for line in axis.lines],
            ["Ground truth", "Sample mean", "Neural-ODE MAP sample"],
        )
        self.assertIn("80% pointwise credible interval", axis.collections[0].get_label())
        plt.close(fig)

    def test_runner_generates_post_run_summary_and_reuses_its_sample_batch(self):
        model = _FakeModel()
        entry = {
            "run_id": "run-0",
            "problem": {"distribution": {"name": "ks"}},
            "method_kwargs": {},
            "resolved_problem": {"dim": 1},
            "architecture": {},
            "checkpoints": {"last": Path("unused")},
        }
        deps = {"nnx": _FakeNNX, "jnp": _FakeJnp}
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(runner, "load_experiment_entries", return_value=[entry]),
                patch.object(runner, "load_entry_model", return_value=model),
                patch.object(runner, "entry_arrays", return_value={}),
                patch.object(runner, "varying_param_keys", return_value=[]),
                patch.object(runner, "format_run_label", return_value="run-0"),
                patch.object(runner, "import_runtime_dependencies", return_value=deps),
                patch.object(runner, "field_truth_for_plot", return_value=np.zeros(1)),
                patch.object(runner, "plot_run_diagnostics") as diagnostics,
            ):
                paths = runner.generate_per_run_plots(
                    temp_dir,
                    {"n_posterior_samples": 3, "posterior_seed": 9},
                )

            self.assertEqual(
                [path.name for path in paths],
                ["run-0_posterior.pdf", "run-0_plots.pdf"],
            )
            self.assertTrue(paths[0].exists())
            self.assertEqual(model.calls, [(1, 9, True)] * 3)
            forwarded_samples = diagnostics.call_args.kwargs["posterior_samples"]
            np.testing.assert_array_equal(forwarded_samples[:, 0], np.arange(3))

if __name__ == "__main__":
    unittest.main()
