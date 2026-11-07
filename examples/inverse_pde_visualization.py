"""Posterior field summaries and plotting helpers for PDE inverse runs.

The public helpers accept explicit samples and neural-ODE log densities, so a
notebook can cache or reuse one draw batch for every summary and figure. The
checkpoint convenience functions draw after training; they do not change the
training loop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _runner_module():
    try:
        from . import benchmark_runner
    except ImportError:
        import benchmark_runner
    return benchmark_runner


def sample_model_posterior(
    model: Any,
    n_samples: int = 256,
    seed: int = 0,
    deps: dict[str, Any] | None = None,
    batch_size: int = 1,
) -> tuple[Any, Any]:
    """Draw fields and their neural-ODE log densities from a loaded model.

    The returned log densities are ``log q_theta(x)`` from the learned
    pushforward, not evaluations of the inverse problem's target posterior.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    runner = _runner_module()
    deps = deps or runner.import_runtime_dependencies()
    rngs = deps["nnx"].Rngs(seed)
    sample_chunks = []
    density_chunks = []
    for start in range(0, n_samples, batch_size):
        chunk_size = min(batch_size, n_samples - start)
        samples, log_density = model.sample(
            chunk_size,
            rngs,
            with_log_density=True,
        )
        sample_chunks.append(samples)
        density_chunks.append(log_density)
    return (
        deps["jnp"].concatenate(sample_chunks, axis=0),
        deps["jnp"].concatenate(density_chunks, axis=0),
    )


def sample_run_posterior(
    session_dir: str | Path,
    run_id: str,
    n_samples: int = 256,
    seed: int = 0,
    checkpoint: str = "last",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Load a run checkpoint and draw fields with neural-ODE log densities.

    Use ``checkpoint='best'`` to select the best checkpoint rather than the
    default final/last checkpoint. The fixed synthetic ground truth is included
    for plotting; it is never used to calculate summaries or choose the MAP
    sample.
    """

    runner = _runner_module()
    if checkpoint not in {"last", "best"}:
        raise ValueError("checkpoint must be either 'last' or 'best'")
    entries = runner.load_experiment_entries(session_dir)
    entry = next((item for item in entries if item.get("run_id") == run_id), None)
    if entry is None:
        raise FileNotFoundError(f"Unknown run_id or missing expanded config: {run_id}")

    field_name = runner._distribution_name(entry["problem"]["distribution"])
    if field_name not in {"acoustic", "ks"}:
        raise ValueError(f"Run {run_id!r} is not an acoustic or KS inverse problem")

    deps = runner.import_runtime_dependencies()
    model = runner.load_entry_model(entry, key=checkpoint)
    samples, log_density = sample_model_posterior(
        model, n_samples=n_samples, seed=seed, deps=deps, batch_size=batch_size
    )
    import numpy as np

    distribution = entry["problem"]["distribution"]
    return {
        "samples": np.asarray(samples),
        "log_density": np.asarray(log_density),
        "truth": np.asarray(runner.field_truth_for_plot(distribution)),
        "distribution": distribution,
        "run_id": run_id,
        "checkpoint": checkpoint,
        "seed": seed,
    }


def summarize_field_samples(
    samples: Any,
    log_density: Any,
    credible_mass: float = 0.8,
) -> dict[str, Any]:
    """Summarize draws; the sampled MAP maximizes neural-ODE log density."""

    import numpy as np

    samples = np.asarray(samples)
    log_density = np.asarray(log_density)
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("samples must have shape (n_samples, n_field_values)")
    if log_density.shape != (samples.shape[0],):
        raise ValueError("log_density must have shape (n_samples,)")
    if not np.isscalar(credible_mass) or not np.isfinite(credible_mass) or not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must be strictly between 0 and 1")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values")
    finite_log_density = np.isfinite(log_density)
    if not np.any(finite_log_density):
        raise ValueError("at least one finite neural-ODE log density is required")

    map_index = int(np.argmax(np.where(finite_log_density, log_density, -np.inf)))
    tail_mass = (1.0 - credible_mass) / 2.0
    return {
        "map_index": map_index,
        "map_sample": samples[map_index],
        "map_log_density": float(log_density[map_index]),
        "mean": np.mean(samples, axis=0),
        "std": np.std(samples, axis=0),
        "lower": np.quantile(samples, tail_mass, axis=0),
        "upper": np.quantile(samples, 1.0 - tail_mass, axis=0),
        "credible_mass": float(credible_mass),
        "n_samples": int(samples.shape[0]),
    }


def plot_acoustic_summary(
    truth: Any,
    summary: dict[str, Any],
    output_path: str | Path | None = None,
):
    """Plot acoustic truth, sampled model-density MAP, mean, and pixel std."""

    import matplotlib.pyplot as plt
    import numpy as np

    truth = np.asarray(truth)
    dim = truth.size
    side = math.isqrt(dim)
    if dim < 1 or side * side != dim:
        raise ValueError("acoustic truth must be a flattened square field")
    fields = [truth, summary["map_sample"], summary["mean"]]
    if (
        any(np.asarray(field).size != dim for field in fields)
        or np.asarray(summary["std"]).size != dim
    ):
        raise ValueError("acoustic summary fields must match the truth grid")
    fields = [np.asarray(field).reshape(side, side) for field in fields]
    std = np.asarray(summary["std"]).reshape(side, side)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    titles = ("Ground truth", "Neural-ODE MAP sample", "Sample mean", "Sample std")
    shared_min = min(float(np.min(field)) for field in fields)
    shared_max = max(float(np.max(field)) for field in fields)
    for axis, field, title in zip(axes[:3], fields, titles[:3]):
        image = axis.imshow(
            field,
            cmap="RdBu_r",
            origin="lower",
            vmin=shared_min,
            vmax=shared_max,
        )
        axis.set_title(title)
        axis.set_xlabel("Pixel column")
        axis.set_ylabel("Pixel row")
        fig.colorbar(image, ax=axis, shrink=0.82)
    image = axes[3].imshow(std, cmap="magma", origin="lower", vmin=0.0)
    axes[3].set_title("Sample std")
    axes[3].set_xlabel("Pixel column")
    axes[3].set_ylabel("Pixel row")
    fig.colorbar(image, ax=axes[3], shrink=0.82)
    if output_path is not None:
        fig.savefig(output_path)
    return fig, axes


def plot_ks_summary(
    truth: Any,
    summary: dict[str, Any],
    output_path: str | Path | None = None,
    domain_length: float = 100.0,
):
    """Plot KS truth, mean, sampled MAP, and pointwise credible interval."""

    import matplotlib.pyplot as plt
    import numpy as np

    truth = np.asarray(truth)
    dim = truth.size
    fields = (summary["map_sample"], summary["mean"], summary["lower"], summary["upper"])
    if dim < 1 or any(np.asarray(field).size != dim for field in fields):
        raise ValueError("KS summary fields must match the truth length")
    if not math.isfinite(domain_length) or domain_length <= 0:
        raise ValueError("domain_length must be a positive finite number")

    x = np.arange(dim) * domain_length / dim
    fig, axis = plt.subplots(1, 1, figsize=(10, 5), layout="constrained")
    axis.fill_between(
        x,
        summary["lower"],
        summary["upper"],
        color="tab:blue",
        alpha=0.22,
        label=f"{summary['credible_mass']:.0%} pointwise credible interval",
        zorder=1,
    )
    axis.plot(x, truth, color="black", linestyle="--", label="Ground truth", zorder=3)
    axis.plot(x, summary["mean"], color="tab:blue", label="Sample mean", zorder=4)
    axis.plot(
        x,
        summary["map_sample"],
        color="tab:orange",
        label="Neural-ODE MAP sample",
        zorder=5,
    )
    axis.set(xlabel="Position", ylabel="Initial condition", title="KS inverse posterior field")
    axis.legend()
    axis.grid(alpha=0.25)
    if output_path is not None:
        fig.savefig(output_path)
    return fig, axis


def plot_run_posterior(
    session_dir: str | Path,
    run_id: str,
    n_samples: int = 256,
    seed: int = 0,
    checkpoint: str = "last",
    output_path: str | Path | None = None,
    batch_size: int = 1,
):
    """Load, draw, summarize, and plot one acoustic or KS inverse run.

    Returns the sampled data, summary, figure, and axes for further notebook
    postprocessing. Supply ``output_path`` to save the figure as well.
    """

    result = sample_run_posterior(
        session_dir,
        run_id,
        n_samples=n_samples,
        seed=seed,
        checkpoint=checkpoint,
        batch_size=batch_size,
    )
    summary = summarize_field_samples(result["samples"], result["log_density"])
    plotter = (
        plot_acoustic_summary
        if result["distribution"]["name"] == "acoustic"
        else plot_ks_summary
    )
    fig, axes = plotter(result["truth"], summary, output_path=output_path)
    return {**result, "summary": summary, "figure": fig, "axes": axes}


__all__ = [
    "sample_model_posterior",
    "sample_run_posterior",
    "summarize_field_samples",
    "plot_acoustic_summary",
    "plot_ks_summary",
    "plot_run_posterior",
]
