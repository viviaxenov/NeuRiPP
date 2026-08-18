import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from neuripp.methods.anderson import get_anderson
from neuripp.methods.ngd import get_ngd, schedule_exp
from neuripp.methods.optax_optimizer import get_optax
from neuripp.parametric_pushforward.flow_matching import FlowMatching, flow_matching_loss
from neuripp.utility.ema import EMA

from rhs_architectures import MLP
from tabular_datasets import (
    EXPECTED_DIMS,
    MAF_ARCHIVE_DOI,
    MAF_ARCHIVE_MD5,
    default_data_dir,
    load_tabular_dataset,
)


DEFAULT_METHODS = "ngd,anderson,adam,sgd"
DEFAULT_N_EPOCHS = "100"
DEFAULT_BATCH_SIZE = 512
DEFAULT_EVAL_EVERY = 10
DEFAULT_EVAL_SAMPLES = 4096
DEFAULT_EVAL_BATCH_SIZE = 256
DEFAULT_TRACE_ESTIMATOR = "exact"
DEFAULT_LOSS_SMOOTHING_ALPHA = 0.05
DEFAULT_SEED = 42
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "flow_matching_tabular_results"
DEFAULT_EMA_DECAY = 0.9999
DEFAULT_EMA_START_STEP = 0

N_PLOT_SAMPLES = 512
N_PCA_FIT_SAMPLES = 10_000
MLP_HIDDEN_DIM = 128
MLP_N_HIDDEN = 3

EVAL_ODE_METHOD = "rk45"
EVAL_ODE_STEPS = 12
EVAL_ODE_KWARGS = {"adaptive": True, "h_max": 0.3}
INFERENCE_ODE_CONFIGS = (
    ("Euler (100 steps)", "euler", 100, {}),
    ("Heun (20 steps)", "heun", 20, {}),
    ("Adaptive RK45 (12 steps)", "rk45", 12, {"adaptive": True, "h_max": 0.3}),
)

NGD_STEP_SIZE = 1e-3
NGD_CLIP_NORM = 20.0
NGD_SOLVER_KWARGS = {
    "linear_solver_regularization": 1e-3,
    "drop_every": 5000,
    "drop_by": 10.0,
    "min_step": 1e-6,
}
ANDERSON_STEP_SIZE = 1e-3
ANDERSON_RELAXATION = 1.0
ANDERSON_REGULARIZATION_FACTOR = 1e-3
ANDERSON_HISTORY_LENGTH = 6
ANDERSON_CLIP_NORM = 20.0
ANDERSON_SOLVER_KWARGS = {
    "linear_solver_regularization": 1e-3,
    "linear_solver_maxiter": 100,
    "drop_every": 5000,
    "drop_by": 10.0,
}
ADAM_LEARNING_RATE = 1e-3
SGD_LEARNING_RATE = 1e-3

SUPPORTED_METHODS = ("ngd", "anderson", "adam", "sgd")


def parse_comma_separated(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train flow-matching models on MAF tabular benchmarks."
    )
    parser.add_argument("dataset", choices=EXPECTED_DIMS)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--n-epochs", "--n_epochs", default=DEFAULT_N_EPOCHS)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-every", "--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--eval-samples", "--eval_samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument(
        "--eval-batch-size",
        "--eval_batch_size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
    )
    parser.add_argument(
        "--trace-estimator",
        "--trace_estimator",
        choices=("exact", "hutchinson"),
        default=DEFAULT_TRACE_ESTIMATOR,
    )
    parser.add_argument(
        "--loss-smoothing-alpha",
        "--loss_smoothing_alpha",
        type=float,
        default=DEFAULT_LOSS_SMOOTHING_ALPHA,
    )
    parser.add_argument("--ema-enabled", "--ema_enabled", action="store_true", default=False)
    parser.add_argument("--ema-decay", "--ema_decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument(
        "--ema-start-step", "--ema_start_step", type=int, default=DEFAULT_EMA_START_STEP
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-dir", "--data_dir", type=Path, default=default_data_dir())
    return parser.parse_args(argv)


def validate_args(args):
    methods = parse_comma_separated(args.methods)
    if not methods:
        raise ValueError("--methods must contain at least one method")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {', '.join(unknown)}")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods must not contain duplicates")

    try:
        n_epochs = [int(value) for value in parse_comma_separated(args.n_epochs)]
    except ValueError as error:
        raise ValueError("--n-epochs must be a comma-separated list of integers") from error
    if len(n_epochs) == 1:
        n_epochs *= len(methods)
    if len(n_epochs) != len(methods):
        raise ValueError("--n-epochs must contain one value or one value per method")
    if any(value < 1 for value in n_epochs):
        raise ValueError("--n-epochs values must be positive")
    if args.batch_size < 2 or args.eval_batch_size < 1 or args.eval_samples < 1:
        raise ValueError("batch sizes and --eval-samples must be positive")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")
    if not 0.0 < args.loss_smoothing_alpha <= 1.0:
        raise ValueError("--loss-smoothing-alpha must be in (0, 1]")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if args.ema_start_step < 0:
        raise ValueError("--ema-start-step must be non-negative")
    return methods, dict(zip(methods, n_epochs, strict=True))


def make_model(seed, batch_size, dim, trace_estimator):
    rngs = nnx.Rngs(seed)
    rhs = MLP(
        dim,
        rngs,
        dim_hidden=MLP_HIDDEN_DIM,
        n_hidden=MLP_N_HIDDEN,
        activation=nnx.swish,
    )
    return FlowMatching(
        rhs,
        rngs,
        batch_size,
        ode_nstep_max=EVAL_ODE_STEPS,
        ode_method=EVAL_ODE_METHOD,
        ode_kwargs=dict(EVAL_ODE_KWARGS),
        divergence_method=trace_estimator,
    )


def make_optimizer(method):
    if method == "ngd":
        init_fn, step_fn = get_ngd(
            flow_matching_loss,
            stepsize_schedule_fn=schedule_exp,
            natural_grad_clipping_threshold=NGD_CLIP_NORM,
        )
        return init_fn, step_fn, (NGD_STEP_SIZE,), dict(NGD_SOLVER_KWARGS)
    if method == "anderson":
        init_fn, step_fn = get_anderson(
            flow_matching_loss,
            stepsize_schedule_fn=schedule_exp,
            history_length=ANDERSON_HISTORY_LENGTH,
            natural_grad_clipping_threshold=ANDERSON_CLIP_NORM,
        )
        optimizer_args = (
            ANDERSON_STEP_SIZE,
            ANDERSON_RELAXATION,
            ANDERSON_REGULARIZATION_FACTOR,
        )
        return init_fn, step_fn, optimizer_args, dict(ANDERSON_SOLVER_KWARGS)
    learning_rate = ADAM_LEARNING_RATE if method == "adam" else SGD_LEARNING_RATE
    init_fn, step_fn = get_optax(flow_matching_loss, method=method)
    return init_fn, step_fn, (learning_rate,), {}


def exponential_smoothing(values, alpha):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def sample_with_solver(model, n_samples, seed, method, n_steps, ode_kwargs):
    previous = (model.ode_method, model.ode_nstep_max, model.ode_kwargs)
    model.ode_method = method
    model.ode_nstep_max = n_steps
    model.ode_kwargs = dict(ode_kwargs)
    try:
        samples = model.sample(n_samples, nnx.Rngs(seed))
        return np.asarray(samples)
    finally:
        model.ode_method, model.ode_nstep_max, model.ode_kwargs = previous


def save_arrays(method_dir, metrics):
    arrays = {name: np.asarray(values) for name, values in metrics.items() if len(values)}
    np.savez(method_dir / "arrays.npz", **arrays)


def fit_pca(data, max_samples=N_PCA_FIT_SAMPLES):
    fit_data = np.asarray(data[:max_samples], dtype=np.float64)
    mean = fit_data.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(fit_data - mean, full_matrices=False)
    return mean, right_vectors[:2].T


def project_pca(data, pca):
    mean, components = pca
    return (np.asarray(data) - mean) @ components


def iter_batches(data, batch_size):
    for start in range(0, len(data), batch_size):
        yield data[start : start + batch_size]


def evaluate_test(model, test_data, train_std, batch_size, seed):
    rngs = nnx.Rngs(seed)
    loss_sum = 0.0
    nll_sum = 0.0
    count = 0
    for batch in iter_batches(test_data, batch_size):
        batch = jnp.asarray(batch)
        batch_count = len(batch)
        loss_sum += float(flow_matching_loss(model, batch, rngs)) * batch_count
        _, log_density = model.pullback(batch, rngs, with_log_density=True)
        nll_sum += float((-log_density).sum())
        count += batch_count

    gaussian_constant = 0.5 * test_data.shape[1] * np.log(2.0 * np.pi)
    nll_standardized = nll_sum / count + gaussian_constant
    nll_original = nll_standardized + np.log(train_std).sum()
    return {
        "test_flow_matching_loss": loss_sum / count,
        "test_nll_standardized": nll_standardized,
        "test_nll_original": nll_original,
    }


def plot_loss(metrics_by_method, output_path, smoothed=False):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    key = "loss_smoothed" if smoothed else "loss"
    for method, metrics in metrics_by_method.items():
        if metrics[key]:
            ax.plot(metrics["step_epoch"], metrics[key], label=method.upper())
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow-matching loss")
    ax.set_title("Exponentially smoothed loss" if smoothed else "Training loss")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def plot_diagnostics(method, epoch, samples, test_data, pca, metrics, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    test_projection = project_pca(test_data[:N_PLOT_SAMPLES], pca)
    sample_projection = project_pca(samples[:N_PLOT_SAMPLES], pca)
    axes[0].scatter(*test_projection.T, s=8, alpha=0.25, label="Test")
    axes[0].scatter(*sample_projection.T, s=8, alpha=0.65, label="Model")
    axes[0].set_title("PCA distribution projection")
    axes[0].legend()

    axes[1].plot(
        metrics["eval_epoch"],
        metrics["test_flow_matching_loss"],
        label="Test flow-matching loss",
    )
    if metrics["ema_test_flow_matching_loss"]:
        axes[1].plot(
            metrics["eval_epoch"],
            metrics["ema_test_flow_matching_loss"],
            linestyle="--",
            color="tab:green",
            label="EMA test flow-matching loss",
        )
    nll_axis = axes[1].twinx()
    nll_axis.plot(
        metrics["eval_epoch"],
        metrics["test_nll_standardized"],
        color="tab:orange",
        label="Test NLL",
    )
    if metrics["ema_test_nll_standardized"]:
        nll_axis.plot(
            metrics["eval_epoch"],
            metrics["ema_test_nll_standardized"],
            linestyle="--",
            color="tab:red",
            label="EMA test NLL",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Test metrics")
    lines = axes[1].lines + nll_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines])

    axes[2].plot(metrics["step_epoch"], metrics["grad_norm"], label="Euclidean gradient")
    if metrics["natural_grad_norm"]:
        axes[2].plot(
            metrics["step_epoch"],
            metrics["natural_grad_norm"],
            label="Natural gradient",
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Gradient norms")
    axes[2].legend()
    fig.suptitle(f"{method.upper()} after epoch {epoch}")
    fig.savefig(output_path)
    plt.close(fig)


def update_plots(
    method,
    epoch,
    model,
    test_data,
    pca,
    metrics,
    all_metrics,
    method_dir,
    output_dir,
    seed,
):
    sampling_start = perf_counter()
    samples = sample_with_solver(
        model,
        N_PLOT_SAMPLES,
        seed + epoch,
        EVAL_ODE_METHOD,
        EVAL_ODE_STEPS,
        EVAL_ODE_KWARGS,
    )
    metrics["sampling_wall_time"].append(perf_counter() - sampling_start)
    plot_start = perf_counter()
    plot_loss({method: metrics}, method_dir / "plots" / "current_loss.pdf")
    plot_loss(
        {method: metrics},
        method_dir / "plots" / "current_loss_smoothed.pdf",
        smoothed=True,
    )
    plot_diagnostics(
        method,
        epoch,
        samples,
        test_data,
        pca,
        metrics,
        method_dir / "plots" / "current_diagnostics.pdf",
    )
    plot_loss(all_metrics, output_dir / "joint_loss.pdf")
    plot_loss(all_metrics, output_dir / "joint_loss_smoothed.pdf", smoothed=True)
    metrics["plot_io_wall_time"].append(perf_counter() - plot_start)
    save_arrays(method_dir, metrics)


def run_evaluation(
    method,
    epoch,
    global_step,
    model,
    evaluation_data,
    diagnostic_data,
    is_full,
    train_std,
    args,
    pca,
    metrics,
    all_metrics,
    method_dir,
    output_dir,
    *,
    ema_model=None,
):
    start = perf_counter()
    values = evaluate_test(
        model,
        evaluation_data,
        train_std,
        args.eval_batch_size,
        args.seed + 50_000,
    )
    metrics["eval_epoch"].append(epoch)
    metrics["eval_step"].append(global_step)
    metrics["eval_is_full"].append(is_full)
    metrics["eval_sample_count"].append(len(evaluation_data))
    for name, value in values.items():
        metrics[name].append(value)
    if ema_model is not None:
        ema_values = evaluate_test(
            ema_model,
            evaluation_data,
            train_std,
            args.eval_batch_size,
            args.seed + 60_000,
        )
        for name, value in ema_values.items():
            metrics[f"ema_{name}"].append(value)
    metrics["eval_wall_time"].append(perf_counter() - start)
    save_arrays(method_dir, metrics)
    update_plots(
        method,
        epoch,
        model,
        diagnostic_data,
        pca,
        metrics,
        all_metrics,
        method_dir,
        output_dir,
        args.seed + 10_000 * (SUPPORTED_METHODS.index(method) + 1),
    )


def train_method(
    method,
    n_epochs,
    args,
    train_data,
    test_data,
    test_subset,
    train_std,
    pca,
    output_dir,
    all_metrics,
):
    method_start = perf_counter()
    method_dir = output_dir / method
    (method_dir / "plots").mkdir(parents=True)
    model = make_model(args.seed, args.batch_size, train_data.shape[1], args.trace_estimator)
    train_rngs = nnx.Rngs(args.seed + 1)
    init_fn, step_fn, optimizer_args, optimizer_kwargs = make_optimizer(method)

    initialization_start = perf_counter()
    initial_batch = jnp.asarray(train_data[: args.batch_size])
    state = init_fn(model, optimizer_args, optimizer_kwargs, initial_batch, train_rngs)
    jax.block_until_ready(state)
    initialization_wall_time = perf_counter() - initialization_start
    step_fn = nnx.jit(step_fn)
    ema = (
        EMA(state[0], decay=args.ema_decay, start_step=args.ema_start_step)
        if args.ema_enabled
        else None
    )
    metrics = {
        "loss": [],
        "loss_smoothed": [],
        "grad_norm": [],
        "natural_grad_norm": [],
        "step": [],
        "step_epoch": [],
        "epoch_train_wall_time": [],
        "cumulative_train_wall_time": [],
        "eval_epoch": [],
        "eval_step": [],
        "eval_is_full": [],
        "eval_sample_count": [],
        "test_flow_matching_loss": [],
        "test_nll_standardized": [],
        "test_nll_original": [],
        "ema_test_flow_matching_loss": [],
        "ema_test_nll_standardized": [],
        "ema_test_nll_original": [],
        "eval_wall_time": [],
        "sampling_wall_time": [],
        "plot_io_wall_time": [],
        "initialization_wall_time": [initialization_wall_time],
        "first_step_wall_time": [],
        "total_wall_time": [],
    }
    all_metrics[method] = metrics
    cumulative_train_time = 0.0
    global_step = 0

    for epoch in tqdm.trange(1, n_epochs + 1, desc=method.upper()):
        permutation = np.random.default_rng(args.seed + epoch).permutation(len(train_data))
        epoch_start = perf_counter()
        n_batches = (len(train_data) + args.batch_size - 1) // args.batch_size
        for batch_index, start in enumerate(range(0, len(train_data), args.batch_size), start=1):
            batch = jnp.asarray(train_data[permutation[start : start + args.batch_size]])
            step_start = perf_counter()
            state, values = step_fn(state, batch, train_rngs)
            loss, grad_norm_sq, *natural_norm_sq = values
            loss_value = float(loss)
            if global_step == 0:
                metrics["first_step_wall_time"].append(perf_counter() - step_start)
            global_step += 1
            if ema is not None:
                ema.update(state[0], global_step)
            metrics["step"].append(global_step)
            metrics["step_epoch"].append(epoch - 1 + batch_index / n_batches)
            metrics["loss"].append(loss_value)
            previous = metrics["loss_smoothed"][-1] if metrics["loss_smoothed"] else loss_value
            metrics["loss_smoothed"].append(
                args.loss_smoothing_alpha * loss_value
                + (1.0 - args.loss_smoothing_alpha) * previous
            )
            metrics["grad_norm"].append(float(jnp.sqrt(jnp.maximum(grad_norm_sq, 0.0))))
            if natural_norm_sq:
                metrics["natural_grad_norm"].append(
                    float(jnp.sqrt(jnp.maximum(natural_norm_sq[0], 0.0)))
                )

        epoch_time = perf_counter() - epoch_start
        cumulative_train_time += epoch_time
        metrics["epoch_train_wall_time"].append(epoch_time)
        metrics["cumulative_train_wall_time"].append(cumulative_train_time)
        if epoch % args.eval_every == 0 and epoch < n_epochs:
            run_evaluation(
                method,
                epoch,
                global_step,
                state[0],
                test_subset,
                test_subset,
                False,
                train_std,
                args,
                pca,
                metrics,
                all_metrics,
                method_dir,
                output_dir,
                ema_model=ema.model if ema is not None else None,
            )

    run_evaluation(
        method,
        n_epochs,
        global_step,
        state[0],
        test_data,
        test_subset,
        True,
        train_std,
        args,
        pca,
        metrics,
        all_metrics,
        method_dir,
        output_dir,
        ema_model=ema.model if ema is not None else None,
    )
    metrics["total_wall_time"].append(perf_counter() - method_start)
    save_arrays(method_dir, metrics)
    if ema is not None:
        params = jax.tree.map(
            lambda value: np.asarray(value), nnx.state(ema.model, nnx.Param)
        )
        np.savez(method_dir / "ema_params.npz", **params)
    return state[0], ema.model if ema is not None else None


def plot_joint_samples(model_rows, test_data, pca, seed, output_path):
    fig, axes = plt.subplots(
        len(model_rows),
        len(INFERENCE_ODE_CONFIGS),
        figsize=(15, 5 * len(model_rows)),
        squeeze=False,
        layout="constrained",
    )
    test_projection = project_pca(test_data[:N_PLOT_SAMPLES], pca)
    for row, (method_name, model) in enumerate(model_rows):
        for col, (title, ode_method, n_steps, ode_kwargs) in enumerate(INFERENCE_ODE_CONFIGS):
            samples = sample_with_solver(
                model, N_PLOT_SAMPLES, seed, ode_method, n_steps, ode_kwargs
            )
            sample_projection = project_pca(samples, pca)
            axes[row, col].scatter(*test_projection.T, s=8, alpha=0.2)
            axes[row, col].scatter(*sample_projection.T, s=8, alpha=0.65)
            if row == 0:
                axes[row, col].set_title(title)
            if col == 0:
                axes[row, col].set_ylabel(method_name.upper())
    fig.savefig(output_path)
    plt.close(fig)


def serializable_config(args, methods, n_epochs, dataset):
    return {
        "dataset": args.dataset,
        "methods": methods,
        "n_epochs": n_epochs,
        "batch_size": args.batch_size,
        "eval_every_epochs": args.eval_every,
        "eval_samples": min(args.eval_samples, len(dataset.test)),
        "eval_batch_size": args.eval_batch_size,
        "trace_estimator": args.trace_estimator,
        "loss_smoothing_alpha": args.loss_smoothing_alpha,
        "ema_enabled": args.ema_enabled,
        "ema_decay": args.ema_decay,
        "ema_start_step": args.ema_start_step,
        "seed": args.seed,
        "train_size": len(dataset.train),
        "test_size": len(dataset.test),
        "dimension": dataset.train.shape[1],
        "data_dir": str(dataset.data_dir),
        "data_source_doi": MAF_ARCHIVE_DOI,
        "data_archive_md5": MAF_ARCHIVE_MD5,
        "download_wall_time": dataset.download_wall_time,
        "preprocessing_wall_time": dataset.preprocessing_wall_time,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "mlp_n_hidden": MLP_N_HIDDEN,
        "ngd_step_size": NGD_STEP_SIZE,
        "anderson_step_size": ANDERSON_STEP_SIZE,
        "adam_learning_rate": ADAM_LEARNING_RATE,
        "sgd_learning_rate": SGD_LEARNING_RATE,
    }


def main(argv=None):
    args = parse_args(argv)
    methods, n_epochs = validate_args(args)
    dataset = load_tabular_dataset(args.dataset, args.data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / args.dataset / timestamp
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(
        json.dumps(serializable_config(args, methods, n_epochs, dataset), indent=2) + "\n"
    )
    evaluation_rng = np.random.default_rng(args.seed + 2)
    evaluation_indices = evaluation_rng.permutation(len(dataset.test))[
        : min(args.eval_samples, len(dataset.test))
    ]
    np.savez(
        output_dir / "normalization.npz",
        train_mean=dataset.train_mean,
        train_std=dataset.train_std,
        evaluation_indices=evaluation_indices,
    )
    np.savez(
        output_dir / "data_timing.npz",
        download_wall_time=np.asarray([dataset.download_wall_time]),
        preprocessing_wall_time=np.asarray([dataset.preprocessing_wall_time]),
    )
    test_subset = dataset.test[evaluation_indices]
    pca = fit_pca(dataset.test)
    model_rows = []
    all_metrics = {}
    for method in methods:
        model, ema_model = train_method(
            method,
            n_epochs[method],
            args,
            dataset.train,
            dataset.test,
            test_subset,
            dataset.train_std,
            pca,
            output_dir,
            all_metrics,
        )
        model_rows.append((method, model))
        if ema_model is not None:
            model_rows.append((f"{method} (EMA)", ema_model))

    plot_loss(all_metrics, output_dir / "joint_loss.pdf")
    plot_loss(all_metrics, output_dir / "joint_loss_smoothed.pdf", smoothed=True)
    plot_joint_samples(
        model_rows,
        dataset.test,
        pca,
        args.seed + 100_000,
        output_dir / "joint_samples.pdf",
    )
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
