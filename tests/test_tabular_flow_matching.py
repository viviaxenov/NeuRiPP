from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import jax.numpy as jnp
import numpy as np
from flax import nnx


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import flow_matching_tabular_training as training
from neuripp.parametric_pushforward.flow_matching import FlowMatching
from tabular_datasets import load_tabular_dataset


def test_miniboone_uses_train_moments_for_test_normalization():
    with tempfile.TemporaryDirectory() as directory:
        data_dir = Path(directory)
        path = data_dir / "extracted" / "miniboone" / "data.npy"
        path.parent.mkdir(parents=True)
        raw = np.arange(100 * 43, dtype=np.float32).reshape(100, 43)
        np.save(path, raw)

        dataset = load_tabular_dataset("miniboone", data_dir)

        raw_train = raw[:81]
        raw_test = raw[90:]
        expected_mean = raw_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        expected_std = raw_train.std(axis=0, dtype=np.float64).astype(np.float32)
        np.testing.assert_allclose(dataset.train_mean, expected_mean)
        np.testing.assert_allclose(dataset.train_std, expected_std)
        np.testing.assert_allclose(dataset.test, (raw_test - expected_mean) / expected_std)


def test_nll_adds_gaussian_constant_and_normalization_jacobian():
    class ZeroRHS(nnx.Module):
        def __init__(self):
            self.dim = 2

        def __call__(self, time, x):
            return jnp.zeros_like(x)

    rngs = nnx.Rngs(0)
    model = FlowMatching(
        ZeroRHS(),
        rngs,
        N_monte_carlo=2,
        ode_nstep_max=2,
        ode_method="euler",
        divergence_method="exact",
    )
    metrics = training.evaluate_test(
        model,
        np.zeros((2, 2), dtype=np.float32),
        np.asarray([2.0, 3.0], dtype=np.float32),
        batch_size=2,
        seed=1,
    )
    np.testing.assert_allclose(metrics["test_nll_standardized"], np.log(2.0 * np.pi))
    np.testing.assert_allclose(
        metrics["test_nll_original"], np.log(2.0 * np.pi) + np.log(6.0)
    )


def test_epoch_training_includes_remainder_and_writes_outputs():
    original_defaults = (
        training.N_PLOT_SAMPLES,
        training.MLP_HIDDEN_DIM,
        training.MLP_N_HIDDEN,
        training.EVAL_ODE_STEPS,
        training.EVAL_ODE_KWARGS,
    )
    training.N_PLOT_SAMPLES = 8
    training.MLP_HIDDEN_DIM = 4
    training.MLP_N_HIDDEN = 1
    training.EVAL_ODE_STEPS = 2
    training.EVAL_ODE_KWARGS = {}
    try:
        args = SimpleNamespace(
            seed=2,
            batch_size=4,
            trace_estimator="exact",
            eval_batch_size=2,
            eval_every=1,
            loss_smoothing_alpha=0.2,
        )
        rng = np.random.default_rng(3)
        train = rng.normal(size=(9, 2)).astype(np.float32)
        test = rng.normal(size=(6, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            metrics = {}
            training.train_method(
                "adam",
                2,
                args,
                train,
                test,
                test[:4],
                np.ones(2, dtype=np.float32),
                training.fit_pca(test),
                output_dir,
                metrics,
            )

            arrays = np.load(output_dir / "adam" / "arrays.npz")
            assert arrays["loss"].shape == (6,)
            assert arrays["loss_smoothed"].shape == (6,)
            assert arrays["epoch_train_wall_time"].shape == (2,)
            assert arrays["eval_epoch"].shape == (2,)
            assert arrays["eval_is_full"].tolist() == [False, True]
            assert np.isfinite(arrays["test_nll_standardized"]).all()
            assert (output_dir / "adam" / "plots" / "current_loss.pdf").is_file()
            assert (
                output_dir / "adam" / "plots" / "current_loss_smoothed.pdf"
            ).is_file()
            assert (
                output_dir / "adam" / "plots" / "current_diagnostics.pdf"
            ).is_file()
            assert (output_dir / "joint_loss.pdf").is_file()
            assert (output_dir / "joint_loss_smoothed.pdf").is_file()
    finally:
        (
            training.N_PLOT_SAMPLES,
            training.MLP_HIDDEN_DIM,
            training.MLP_N_HIDDEN,
            training.EVAL_ODE_STEPS,
            training.EVAL_ODE_KWARGS,
        ) = original_defaults


if __name__ == "__main__":
    test_miniboone_uses_train_moments_for_test_normalization()
    test_nll_adds_gaussian_constant_and_normalization_jacobian()
    test_epoch_training_includes_remainder_and_writes_outputs()
    print("Tabular flow-matching tests passed.")
