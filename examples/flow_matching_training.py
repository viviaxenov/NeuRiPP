import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
from datetime import datetime
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from neuripp.functionals.MMD import bandwidth_median, gaussian_mmd
from neuripp.methods.anderson import get_anderson
from neuripp.methods.ngd import get_ngd, schedule_exp
from neuripp.methods.optax_optimizer import get_optax
from neuripp.parametric_pushforward.flow_matching import FlowMatching, flow_matching_loss

from data_generators import CheckerboardBatcher, EightGaussiansBatcher, TwoSpiralsBatcher
from rhs_architectures import MLP

gaussian_mmd = nnx.jit(gaussian_mmd)


DEFAULT_TARGET = "checkerboard"
DEFAULT_METHODS = "ngd,anderson,adam,sgd"
DEFAULT_N_STEPS = "6000"
DEFAULT_BATCH_SIZE = 512
DEFAULT_EVAL_EVERY = 100
DEFAULT_SEED = 42
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "flow_matching_results"

DIM = 2
RESAMPLE_EACH = 1
N_VALIDATION_SAMPLES = 512
N_PLOT_SAMPLES = 512
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
    "min_step": 1e-6
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
TARGET_BATCHERS = {
    "checkerboard": CheckerboardBatcher,
    "eight_gaussians": EightGaussiansBatcher,
    "two_spirals": TwoSpiralsBatcher,
}


def parse_comma_separated(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train flow-matching models on toy data.")
    parser.add_argument("--target", choices=TARGET_BATCHERS, default=DEFAULT_TARGET)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--n-steps", "--n_steps", default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-every", "--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
        n_steps = [int(value) for value in parse_comma_separated(args.n_steps)]
    except ValueError as error:
        raise ValueError("--n-steps must be a comma-separated list of integers") from error
    if len(n_steps) == 1:
        n_steps *= len(methods)
    if len(n_steps) != len(methods):
        raise ValueError("--n-steps must contain one value or one value per method")
    if any(value < 1 for value in n_steps):
        raise ValueError("--n-steps values must be positive")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")
    return methods, dict(zip(methods, n_steps, strict=True))


def make_model(seed, batch_size):
    rngs = nnx.Rngs(seed)
    rhs = MLP(
        DIM,
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


def sample_with_solver(model, n_samples, seed, method, n_steps, ode_kwargs):
    previous = (model.ode_method, model.ode_nstep_max, model.ode_kwargs)
    model.ode_method = method
    model.ode_nstep_max = n_steps
    model.ode_kwargs = dict(ode_kwargs)
    try:
        return model.sample(n_samples, nnx.Rngs(seed))
    finally:
        model.ode_method, model.ode_nstep_max, model.ode_kwargs = previous


def save_arrays(method_dir, metrics):
    arrays = {name: np.asarray(values) for name, values in metrics.items() if values}
    np.savez(method_dir / "arrays.npz", **arrays)


def plot_method(method, iteration, samples, validation_data, metrics, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")

    axes[0].scatter(*np.asarray(validation_data).T, s=8, alpha=0.25, label="Target")
    axes[0].scatter(*np.asarray(samples).T, s=8, alpha=0.65, label="Model")
    axes[0].set_title("Distribution samples")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()

    loss_iterations = np.arange(1, len(metrics["loss"]) + 1)
    axes[1].plot(loss_iterations, metrics["loss"], label="Training loss")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Loss and validation MMD")
    mmd_axis = axes[1].twinx()
    mmd_axis.plot(
        metrics["eval_iteration"], metrics["validation_mmd"], color="tab:orange", label="Validation MMD"
    )
    mmd_axis.set_ylabel("MMD")
    lines = axes[1].lines + mmd_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines])

    axes[2].plot(loss_iterations, metrics["grad_norm"], label="Euclidean gradient")
    if metrics["natural_grad_norm"]:
        axes[2].plot(
            loss_iterations,
            metrics["natural_grad_norm"],
            label="Natural gradient",
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("Gradient norms")
    axes[2].legend()

    fig.suptitle(f"{method.upper()} after {iteration} iterations")
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_and_save(
    method,
    iteration,
    model,
    validation_data,
    bandwidth,
    metrics,
    method_dir,
    seed,
):
    samples = sample_with_solver(
        model,
        max(N_VALIDATION_SAMPLES, N_PLOT_SAMPLES),
        seed + iteration,
        EVAL_ODE_METHOD,
        EVAL_ODE_STEPS,
        EVAL_ODE_KWARGS,
    )
    mmd = gaussian_mmd(
        samples[:N_VALIDATION_SAMPLES],
        validation_data,
        jnp.asarray([bandwidth]),
    )
    metrics["eval_iteration"].append(iteration)
    metrics["validation_mmd"].append(float(mmd))
    save_arrays(method_dir, metrics)
    plot_method(
        method,
        iteration,
        samples[:N_PLOT_SAMPLES],
        validation_data[:N_PLOT_SAMPLES],
        metrics,
        method_dir / "plots" / f"last.pdf",
    )


def train_method(method, n_steps, args, validation_data, bandwidth, output_dir):
    method_dir = output_dir / method
    (method_dir / "plots").mkdir(parents=True)
    model = make_model(args.seed, args.batch_size)
    train_rngs = nnx.Rngs(args.seed + 1)
    batcher = TARGET_BATCHERS[args.target](args.batch_size, RESAMPLE_EACH)
    init_fn, step_fn, optimizer_args, optimizer_kwargs = make_optimizer(method)
    initial_batch = batcher(train_rngs)
    state = init_fn(model, optimizer_args, optimizer_kwargs, initial_batch, train_rngs)
    step_fn = nnx.jit(step_fn)
    metrics = {
        "loss": [],
        "grad_norm": [],
        "natural_grad_norm": [],
        "eval_iteration": [],
        "validation_mmd": [],
    }

    for iteration in tqdm.trange(1, n_steps + 1, desc=method.upper()):
        batch = batcher(train_rngs)
        state, values = step_fn(state, batch, train_rngs)
        loss, grad_norm_sq, *natural_norm_sq = values
        metrics["loss"].append(float(loss))
        metrics["grad_norm"].append(float(jnp.sqrt(jnp.maximum(grad_norm_sq, 0.0))))
        if natural_norm_sq:
            metrics["natural_grad_norm"].append(
                float(jnp.sqrt(jnp.maximum(natural_norm_sq[0], 0.0)))
            )

        if iteration % args.eval_every == 0 or iteration == n_steps:
            evaluate_and_save(
                method,
                iteration,
                state[0],
                validation_data,
                bandwidth,
                metrics,
                method_dir,
                args.seed + 10_000 * (SUPPORTED_METHODS.index(method) + 1),
            )

    return state[0], metrics


def plot_joint_loss(all_metrics, output_path):
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for method, metrics in all_metrics.items():
        ax.plot(np.arange(1, len(metrics["loss"]) + 1), metrics["loss"], label=method.upper())
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Flow-matching loss")
    ax.set_title("Flow-matching training comparison")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)


def plot_joint_samples(models, validation_data, seed, output_path):
    n_rows = len(models)
    n_cols = len(INFERENCE_ODE_CONFIGS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    for row, (method_name, model) in enumerate(models.items()):
        for col, (title, ode_method, n_steps, ode_kwargs) in enumerate(INFERENCE_ODE_CONFIGS):
            samples = sample_with_solver(
                model,
                N_PLOT_SAMPLES,
                seed,
                ode_method,
                n_steps,
                ode_kwargs,
            )
            ax = axes[row, col]
            ax.scatter(*np.asarray(validation_data[:N_PLOT_SAMPLES]).T, s=8, alpha=0.2)
            ax.scatter(*np.asarray(samples).T, s=8, alpha=0.65)
            ax.set_aspect("equal", adjustable="datalim")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(method_name.upper())
    fig.savefig(output_path)
    plt.close(fig)


def serializable_config(args, methods, n_steps):
    return {
        "target": args.target,
        "methods": methods,
        "n_steps": n_steps,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "validation_samples": N_VALIDATION_SAMPLES,
        "plot_samples": N_PLOT_SAMPLES,
        "mlp_hidden_dim": MLP_HIDDEN_DIM,
        "mlp_n_hidden": MLP_N_HIDDEN,
        "ngd_step_size": NGD_STEP_SIZE,
        "anderson_step_size": ANDERSON_STEP_SIZE,
        "adam_learning_rate": ADAM_LEARNING_RATE,
        "sgd_learning_rate": SGD_LEARNING_RATE,
    }


def main(argv=None):
    args = parse_args(argv)
    methods, n_steps = validate_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / args.target / timestamp
    output_dir.mkdir(parents=True)

    config = serializable_config(args, methods, n_steps)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    validation_rngs = nnx.Rngs(args.seed + 2)
    validation_batcher = TARGET_BATCHERS[args.target](N_VALIDATION_SAMPLES, 1)
    validation_data = validation_batcher(validation_rngs)
    bandwidth = bandwidth_median(validation_data)

    models = {}
    all_metrics = {}
    for method in methods:
        model, metrics = train_method(
            method,
            n_steps[method],
            args,
            validation_data,
            bandwidth,
            output_dir,
        )
        models[method] = model
        all_metrics[method] = metrics

    plot_joint_loss(all_metrics, output_dir / "joint_loss.pdf")
    plot_joint_samples(models, validation_data, args.seed + 100_000, output_dir / "joint_samples.pdf")
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
