"""Benchmark runner CLI skeleton.

This module intentionally avoids importing JAX at module import time. Worker
execution will set device-related environment variables before JAX is loaded in
later implementation stages.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import itertools
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import traceback
import warnings
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "output_root",
    "common_params",
    "problem",
    "architectures",
    "methods",
    "plotting",
}
OPTIONAL_TOP_LEVEL_KEYS = {"parallel", "config"}
SUPPORTED_FUNCTIONAL_KINDS = {"KL", "MMD", "CrossEntropy"}
DATA_DISTRIBUTIONS = {"checkerboard", "two_spirals", "eight_gaussians"}
IMAGE_DATA_DISTRIBUTIONS = {"mnist", "fashion_mnist"}
ANALYTIC_KL_DISTRIBUTIONS = {"gaussian", "st"}

_IMAGE_DATASET_CACHE: dict[str, dict[str, Any]] = {}

# Stage 3 validates method names without importing JAX/Optax at module import time.
# Later stages will replace these placeholders with the real dispatch callables.
str_to_method = {
    "sgd": None,
    "ngd": None,
    "anderson": None,
    "adam": None,
    "adamw": None,
    "rmsprop": None,
    "adagrad": None,
    "yogi": None,
    "lion": None,
    "lbfgs": None,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a JSON object")
    return data


def _load_json_value(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file {path}: {exc}") from exc


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_non_empty_object_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{index}] must be an object")
    return value


def _validate_grid_axes(template: dict[str, Any], name: str) -> None:
    for key, value in template.items():
        if isinstance(value, list) and not value:
            raise ValueError(f"{name}.{key} must not be an empty list")


def _validate_architecture(architecture: dict[str, Any], index: int) -> None:
    name = f"architectures[{index}]"
    if "dim" in architecture:
        raise ValueError(f"{name}.dim is not supported; dim is inferred from problem")

    rhs = _require_object(architecture.get("rhs"), f"{name}.rhs")
    if "dim" in rhs:
        raise ValueError(
            f"{name}.rhs.dim is not supported; dim is inferred from problem"
        )
    if not isinstance(rhs.get("model"), str):
        raise ValueError(f"{name}.rhs.model must be a string")

    _validate_grid_axes(
        {key: value for key, value in architecture.items() if key != "rhs"}, name
    )
    _validate_grid_axes(rhs, f"{name}.rhs")


def _distribution_name(distribution: dict[str, Any]) -> str:
    name = distribution.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("problem.distribution.name must be a non-empty string")
    return name


def _is_image_distribution(name: str) -> bool:
    return name in IMAGE_DATA_DISTRIBUTIONS


def _image_dataset_hf_name(name: str) -> str:
    if name == "mnist":
        return "mnist"
    if name == "fashion_mnist":
        return "zalando-datasets/fashion_mnist"
    raise ValueError(f"Unsupported image dataset {name!r}")


def load_image_dataset(name: str) -> dict[str, Any]:
    import numpy as np
    from datasets import load_dataset

    if name in _IMAGE_DATASET_CACHE:
        return _IMAGE_DATASET_CACHE[name]

    dataset = load_dataset(_image_dataset_hf_name(name))

    def _stack_images(split_name: str) -> np.ndarray:
        return (
            np.stack(
                [np.asarray(image, dtype=np.float32) for image in dataset[split_name]["image"]],
                axis=0,
            )
            / 255.0
        )

    train_images = _stack_images("train")
    test_images = _stack_images("test")
    mean = train_images.mean(axis=0)
    std = train_images.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)

    train_flat = ((train_images - mean[None, ...]) / std[None, ...]).reshape(
        train_images.shape[0], -1
    )
    test_flat = ((test_images - mean[None, ...]) / std[None, ...]).reshape(
        test_images.shape[0], -1
    )
    log_det = float(np.log(255.0 * std.reshape(-1)).sum(dtype=np.float64))

    payload = {
        "train": train_flat,
        "test": test_flat,
        "mean": mean,
        "std": std,
        "image_shape": tuple(mean.shape),
        "pixel_log_det_per_example": log_det,
    }
    _IMAGE_DATASET_CACHE[name] = payload
    return payload


def _sample_data_distribution(
    distribution: dict[str, Any], common_params: dict[str, Any]
) -> Any:
    import numpy as np

    name = _distribution_name(distribution)
    sample_count = common_params.get("batch_size", 16)
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("common_params.batch_size must be a positive integer")
    seed = distribution.get("seed", 3)
    rng = np.random.default_rng(seed)

    if _is_image_distribution(name):
        dataset = load_image_dataset(name)
        train = dataset["train"]
        if sample_count > train.shape[0]:
            raise ValueError(
                f"common_params.batch_size={sample_count} exceeds dataset size for {name!r}"
            )
        indices = rng.choice(train.shape[0], size=sample_count, replace=False)
        return train[indices]

    if name == "checkerboard":
        points = rng.uniform(size=(sample_count, 2))
        shifts_x = rng.integers(0, 4, size=sample_count) - 2
        shifts_y = rng.integers(0, 2, size=sample_count) * 2 + shifts_x % 2 - 2
        points[:, 0] += shifts_x
        points[:, 1] += shifts_y
        return points

    if name == "two_spirals":
        half = max(sample_count // 2, 1)
        n = np.sqrt(rng.uniform(size=(half, 1))) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.uniform(size=(half, 1)) * 0.5
        d1y = np.sin(n) * n + rng.uniform(size=(half, 1)) * 0.5
        points = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        points += rng.uniform(size=points.shape) * 0.1
        return points[:sample_count]

    if name == "eight_gaussians":
        theta = np.linspace(0.0, 2.0 * np.pi, 8)
        centers = 4.0 * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
        blob = rng.normal(size=(sample_count, 2)) * 0.5
        shift_ids = rng.integers(0, 7, size=sample_count)
        return (blob + centers[shift_ids, :]) / 1.414

    supported = ", ".join(sorted(DATA_DISTRIBUTIONS | IMAGE_DATA_DISTRIBUTIONS))
    raise ValueError(
        f"Unsupported data distribution {name!r}. Supported data distributions: {supported}"
    )


def resolve_problem(
    problem: dict[str, Any], common_params: dict[str, Any]
) -> dict[str, Any]:
    """Resolve problem metadata that execution can trust later.

    Data-backed distributions infer dimension once here. Analytic KL targets use
    the configured problem dimension.
    """

    functional = problem["functional"]
    kind = functional["kind"]
    distribution = problem["distribution"]
    name = _distribution_name(distribution)
    configured_dim = problem.get("dim")
    if configured_dim is not None and (
        not isinstance(configured_dim, int) or configured_dim < 1
    ):
        raise ValueError("problem.dim must be a positive integer when provided")

    if name in DATA_DISTRIBUTIONS | IMAGE_DATA_DISTRIBUTIONS:
        data_batch = _sample_data_distribution(distribution, common_params)
        if len(data_batch.shape) < 2:
            raise ValueError(
                f"Data distribution {name!r} produced data with invalid shape {data_batch.shape}"
            )
        inferred_dim = int(data_batch.shape[-1])
        if configured_dim is not None and configured_dim != inferred_dim:
            warnings.warn(
                f"problem.dim={configured_dim} does not match data-inferred "
                f"dim={inferred_dim} for distribution {name!r}; using data-inferred dim.",
                stacklevel=2,
            )
        return {
            "dim": inferred_dim,
            "dim_source": "data",
            "distribution_name": name,
            **({"rhs_dim": load_image_dataset(name)["image_shape"]} if _is_image_distribution(name) else {}),
        }

    if kind == "KL" and name in ANALYTIC_KL_DISTRIBUTIONS:
        if configured_dim is None:
            raise ValueError(f"problem.dim is required for KL distribution {name!r}")
        return {
            "dim": configured_dim,
            "dim_source": "config",
            "distribution_name": name,
        }

    if kind in {"CrossEntropy", "MMD"}:
        supported = ", ".join(sorted(DATA_DISTRIBUTIONS | IMAGE_DATA_DISTRIBUTIONS))
        raise ValueError(
            f"Functional {kind} requires a data-backed distribution; got {name!r}. "
            f"Supported data distributions: {supported}"
        )

    if kind == "KL":
        supported = ", ".join(sorted(ANALYTIC_KL_DISTRIBUTIONS))
        raise ValueError(
            f"Unsupported KL distribution {name!r}. Supported analytic distributions: {supported}"
        )

    raise ValueError(f"Unsupported functional kind {kind!r}")


def _expand_template(
    template: dict[str, Any], skip_keys: set[str] | None = None
) -> list[dict[str, Any]]:
    skip_keys = skip_keys or set()
    scalar_items: list[tuple[str, Any]] = []
    grid_items: list[tuple[str, list[Any]]] = []

    for key, value in template.items():
        if key in skip_keys:
            continue
        if isinstance(value, list):
            grid_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if not grid_items:
        return [{key: copy.deepcopy(value) for key, value in scalar_items}]

    expanded = []
    grid_keys = [key for key, _ in grid_items]
    grid_values = [values for _, values in grid_items]
    for combination in itertools.product(*grid_values):
        item = {key: copy.deepcopy(value) for key, value in scalar_items}
        for key, value in zip(grid_keys, combination, strict=True):
            item[key] = copy.deepcopy(value)
        expanded.append(item)
    return expanded


def _expand_architectures(architectures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for architecture in architectures:
        top_level = {key: value for key, value in architecture.items() if key != "rhs"}
        expanded_top_level = _expand_template(top_level)
        expanded_rhs = _expand_template(architecture["rhs"])
        for top_values, rhs_values in itertools.product(
            expanded_top_level, expanded_rhs
        ):
            expanded_architecture = {}
            for key in architecture:
                if key == "rhs":
                    expanded_architecture["rhs"] = copy.deepcopy(rhs_values)
                else:
                    expanded_architecture[key] = copy.deepcopy(top_values[key])
            expanded.append(expanded_architecture)
    return expanded


def _expand_methods(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for method in methods:
        method_name = method["method"]
        n_restarts = method["n_restarts"]
        for kwargs in _expand_template(method, skip_keys={"method", "n_restarts"}):
            for restart_index in range(n_restarts):
                expanded.append(
                    {
                        "method": method_name,
                        "method_kwargs": kwargs,
                        "restart_index": restart_index,
                    }
                )
    return expanded


def _expand_problem(problem: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep a single resolved problem definition for all planned runs."""
    distribution = problem.get("distribution")
    if not isinstance(distribution, dict):
        return [copy.deepcopy(problem)]

    seed_value = distribution.get("seed")
    if not isinstance(seed_value, list):
        return [copy.deepcopy(problem)]

    raise ValueError(
        "problem.distribution.seed must be a scalar; use methods[].n_restarts with "
        "common_params.master_seed for repeated runs"
    )


def plan_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a normalized config into deterministic planned run records."""

    expanded_architectures = _expand_architectures(config["architectures"])
    expanded_methods = _expand_methods(config["methods"])
    expanded_problems = _expand_problem(config["problem"])
    common_params = config["common_params"]
    planned_runs: list[dict[str, Any]] = []

    for architecture, method_config, problem in itertools.product(
        expanded_architectures, expanded_methods, expanded_problems
    ):
        run_index = len(planned_runs)
        method_kwargs = {
            **copy.deepcopy(common_params),
            **copy.deepcopy(method_config["method_kwargs"]),
        }
        planned_runs.append(
            {
                "run_index": run_index,
                "run_id": f"run_{run_index:04d}",
                "restart_index": method_config["restart_index"],
                "problem": copy.deepcopy(problem),
                "resolved_problem": copy.deepcopy(config["resolved_problem"]),
                "architecture": copy.deepcopy(architecture),
                "method": method_config["method"],
                "method_kwargs": method_kwargs,
            }
        )

    return planned_runs


def _method_factory_group_fields(
    method_name: str, method_kwargs: dict[str, Any], deps: dict[str, Any] | None = None
) -> tuple[tuple[str, str], ...]:
    factory_kwargs, _, _ = split_method_kwargs(method_name, method_kwargs, deps=deps)
    return tuple(
        (key, _value_identity(factory_kwargs[key])) for key in sorted(factory_kwargs)
    )


def execution_group_key(
    planned_run: dict[str, Any],
) -> tuple[str, str, tuple[Any, ...]]:
    method = planned_run["method"]
    architecture_key = _value_identity(planned_run["architecture"])
    return (
        architecture_key,
        method,
        _method_factory_group_fields(method, planned_run["method_kwargs"]),
    )


def chunk_planned_runs(
    selected_runs: list[dict[str, Any]], parallel_config: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped_runs: dict[tuple[str, str, tuple[Any, ...]], list[dict[str, Any]]] = {}
    ordered_keys: list[tuple[str, str, tuple[Any, ...]]] = []
    max_parallel = parallel_config["max_parallel"]

    for planned_run in selected_runs:
        group_key = execution_group_key(planned_run)
        if group_key not in grouped_runs:
            grouped_runs[group_key] = []
            ordered_keys.append(group_key)
        grouped_runs[group_key].append(planned_run)

    chunks: list[dict[str, Any]] = []
    for group_key in ordered_keys:
        group_runs = grouped_runs[group_key]
        method = group_runs[0]["method"]
        chunk_size = max_parallel[method]
        for start in range(0, len(group_runs), chunk_size):
            chunk_runs = group_runs[start : start + chunk_size]
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "method": method,
                    "group_key": group_key,
                    "planned_runs": chunk_runs,
                }
            )

    return chunks


def select_planned_runs(
    planned_runs: list[dict[str, Any]], run_id: int | None
) -> list[dict[str, Any]]:
    if run_id is None:
        return planned_runs

    if run_id < 0 or run_id >= len(planned_runs):
        raise ValueError(
            f"--run-id {run_id} is outside the valid range 0..{len(planned_runs) - 1}"
        )
    return [planned_runs[run_id]]


def _sanitize_run_name(run_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip())
    sanitized = sanitized.strip("._")
    if not sanitized:
        raise ValueError("--run-name must contain at least one path-safe character")
    return sanitized


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
        f.write("\n")


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def write_expanded_run_config(run_dir: Path, planned_run: dict[str, Any]) -> None:
    _write_json(run_dir / "expanded_config.json", planned_run)


def write_status(
    run_dir: Path,
    status: str,
    planned_run: dict[str, Any],
    worker_id: int,
    device_info: dict[str, Any],
    **extra: Any,
) -> None:
    payload = {
        "status": status,
        "run_id": planned_run["run_id"],
        "worker_id": worker_id,
        "updated_at": _utc_now_iso(),
        **device_info,
        **extra,
    }
    _write_json(run_dir / "status.json", payload)


def write_error_file(
    run_dir: Path,
    planned_run: dict[str, Any],
    worker_id: int,
    error: BaseException,
    traceback_text: str,
) -> None:
    text = (
        f"Run failed: {planned_run['run_id']}\n"
        f"Worker: {worker_id}\n\n"
        f"Error:\n{error}\n\n"
        f"Traceback:\n{traceback_text}"
    )
    (run_dir / "error.txt").write_text(text, encoding="utf-8")


def _to_float(value: Any) -> float:
    import numpy as np

    return float(np.asarray(value))


def _append_metrics(metric_values: Any, metrics: dict[str, list[float]]) -> None:
    names = ["loss", "grad_norm", "natural_grad_norm"]
    for name, value in zip(names, metric_values, strict=False):
        metrics[name].append(_to_float(value))


def _metric_arrays(metrics: dict[str, list[float]]) -> dict[str, Any]:
    import numpy as np

    return {
        name: np.asarray(values, dtype=float)
        for name, values in metrics.items()
        if values
    }


def _model_from_carry(carry: Any) -> Any:
    return carry[0]


def write_checkpoint(
    checkpoint_dir: Path,
    model: Any,
    deps: dict[str, Any],
    planned_run: dict[str, Any],
    key: str,
    metric_value: float | None = None,
) -> None:
    import orbax.checkpoint as ocp

    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    _, state = deps["nnx"].split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_dir, state)
    checkpointer.wait_until_finished()

    metadata = {
        "run_id": planned_run["run_id"],
        "checkpoint": key,
        "format": "orbax.checkpoint.StandardCheckpointer(nnx_state)",
        "best_checkpoint_criterion": "minimum_loss",
        "written_at": _utc_now_iso(),
    }
    if metric_value is not None:
        metadata["metric_value"] = metric_value
    _write_json(checkpoint_dir / "metadata.json", metadata)


def _latest_session(output_root: Path) -> Path:
    if not output_root.exists():
        raise FileNotFoundError(f"No benchmark output root exists: {output_root}")
    sessions = [path for path in output_root.iterdir() if path.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"No benchmark sessions found under: {output_root}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def resolve_session_dir(
    config: dict[str, Any],
    run_name: str | None,
    output_dir: str | None,
    plot_only: bool,
) -> Path:
    if output_dir is not None:
        session_dir = Path(output_dir)
    else:
        output_root = Path(config["output_root"])
        if run_name is not None:
            session_dir = output_root / _sanitize_run_name(run_name)
        elif plot_only:
            session_dir = _latest_session(output_root)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = output_root / f"benchmark_{timestamp}"

    if plot_only and not session_dir.exists():
        raise FileNotFoundError(f"Benchmark session does not exist: {session_dir}")
    return session_dir


def initialize_session(
    session_dir: Path,
    input_config_path: Path,
    normalized_config: dict[str, Any],
    planned_runs: list[dict[str, Any]],
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "runs").mkdir(exist_ok=True)
    (session_dir / "plots").mkdir(exist_ok=True)

    input_text = input_config_path.read_text(encoding="utf-8")
    (session_dir / "input_config.json").write_text(input_text, encoding="utf-8")
    _write_json(session_dir / "normalized_config.json", normalized_config)
    _write_json(session_dir / "planned_runs.json", planned_runs)


def assigned_gpu_id(
    parallel_config: dict[str, Any], worker_id: int
) -> int | str | None:
    gpu_ids = parallel_config.get("gpu_ids")
    if not gpu_ids:
        return None
    return gpu_ids[worker_id % len(gpu_ids)]


def setup_worker_environment(
    parallel_config: dict[str, Any], worker_id: int
) -> int | str | None:
    """Set worker environment before JAX is imported."""

    requested_gpu_id = assigned_gpu_id(parallel_config, worker_id)
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEME_FRACTION"] = ".90"
    if requested_gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu_id)
    return requested_gpu_id


def apply_jax_config_updates(jax: Any, jax_config: list[list[Any]]) -> None:
    """Apply JAX config flag/value pairs after importing JAX."""
    for flag, value in jax_config:
        jax.config.update(flag, value)


def get_worker_device_info(
    requested_gpu_id: int | str | None,
    jax_config: list[list[Any]] | None = None,
) -> dict[str, Any]:
    """Import JAX after environment setup, apply config, and report actual worker devices."""

    import jax

    if jax_config:
        apply_jax_config_updates(jax, jax_config)
    devices = jax.devices()
    device_strings = [str(device) for device in devices]
    platforms = {getattr(device, "platform", "unknown").lower() for device in devices}
    running_on_gpu = bool(platforms & {"gpu", "cuda", "rocm"})
    device_label = f"GPU {requested_gpu_id}" if running_on_gpu else "CPU"
    return {
        "requested_gpu_id": requested_gpu_id,
        "device_label": device_label,
        "jax_devices": device_strings,
        "jax_platforms": sorted(platforms),
    }


def import_runtime_dependencies() -> dict[str, Any]:
    """Import JAX-dependent objects after worker environment setup."""

    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    import jax
    import jax.numpy as jnp
    from flax import nnx

    from neuripp.functionals.CrossEntropy import cross_entropy
    from neuripp.functionals.KL import getKL
    from neuripp.functionals.MMD import getMMD
    from neuripp.methods.anderson import get_anderson
    from neuripp.methods.ngd import get_ngd
    from neuripp.methods.optax_optimizer import get_optax, optax_optimizers
    from neuripp.parametric_pushforward.parametric_pushforward import (
        ParametricPushforward,
    )

    logpdf_targets = import_module("logpdf_targets")
    rhs_architectures = import_module("rhs_architectures")
    data_generators = import_module("data_generators")

    activation_registry = {
        "tanh": nnx.tanh,
        "swish": nnx.swish,
        "selu": nnx.selu,
    }
    if hasattr(nnx, "relu"):
        activation_registry["relu"] = nnx.relu
    else:
        activation_registry["relu"] = jax.nn.relu
    if hasattr(nnx, "gelu"):
        activation_registry["gelu"] = nnx.gelu
    else:
        activation_registry["gelu"] = jax.nn.gelu

    return {
        "jax": jax,
        "jnp": jnp,
        "nnx": nnx,
        "ParametricPushforward": ParametricPushforward,
        "cross_entropy": cross_entropy,
        "getKL": getKL,
        "logpdf_st": logpdf_targets.logpdf_st,
        "getMMD": getMMD,
        "data_batchers": {
            "checkerboard": data_generators.CheckerboardBatcher,
            "two_spirals": data_generators.TwoSpiralsBatcher,
            "eight_gaussians": data_generators.EightGaussiansBatcher,
        },
        "DatasetBatcher": data_generators.DatasetBatcher,
        "LatentBatcherFromModel": data_generators.LatentBatcherFromModel,
        "ZipBatcher": data_generators.ZipBatcher,
        "rhs_registry": {
            "MLP": rhs_architectures.MLP,
            "LinearRHS": rhs_architectures.LinearRHS,
            "CFMConv2D": rhs_architectures.CFMConv2D,
        },
        "activation_registry": activation_registry,
        "method_registry": {
            "ngd": get_ngd,
            "anderson": get_anderson,
            **{name: get_optax for name in optax_optimizers},
        },
        "optax_methods": set(optax_optimizers),
    }


def build_rhs(
    architecture: dict[str, Any],
    dim: int,
    rngs: Any,
    deps: dict[str, Any] | None = None,
) -> Any:
    deps = deps or import_runtime_dependencies()
    rhs_config = architecture["rhs"]
    rhs_model_name = rhs_config["model"]
    rhs_registry = deps["rhs_registry"]
    if rhs_model_name not in rhs_registry:
        supported = ", ".join(sorted(rhs_registry))
        raise ValueError(
            f"Unsupported RHS model {rhs_model_name!r}; expected one of: {supported}"
        )

    rhs_kwargs = {
        key: copy.deepcopy(value) for key, value in rhs_config.items() if key != "model"
    }
    if isinstance(rhs_kwargs.get("activation"), str):
        activation_name = rhs_kwargs["activation"]
        activation_registry = deps["activation_registry"]
        if activation_name not in activation_registry:
            supported = ", ".join(sorted(activation_registry))
            raise ValueError(
                f"Unsupported activation {activation_name!r}; expected one of: {supported}"
            )
        rhs_kwargs["activation"] = activation_registry[activation_name]

    return rhs_registry[rhs_model_name](dim, rngs=rngs, **rhs_kwargs)


def split_architecture_kwargs(
    architecture: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rhs_config = architecture["rhs"]
    direct_keys = {
        "rhs",
        "N_monte_carlo",
        "seed",
        "ode_nstep_max",
        "ode_method",
        "divergence_method",
    }
    ode_kwargs = {
        key: copy.deepcopy(value)
        for key, value in architecture.items()
        if key not in direct_keys
    }
    return rhs_config, ode_kwargs


def build_model(
    architecture: dict[str, Any],
    method_kwargs: dict[str, Any],
    dim: int,
    rngs: Any,
    deps: dict[str, Any] | None = None,
) -> Any:
    deps = deps or import_runtime_dependencies()
    _, ode_kwargs = split_architecture_kwargs(architecture)
    rhs = build_rhs(architecture, dim, rngs=rngs, deps=deps)
    n_monte_carlo = architecture.get(
        "N_monte_carlo",
        method_kwargs.get("N_monte_carlo", method_kwargs.get("batch_size")),
    )
    if not isinstance(n_monte_carlo, int) or n_monte_carlo < 1:
        raise ValueError("N_monte_carlo must be a positive integer")
    return deps["ParametricPushforward"](
        rhs,
        rngs,
        n_monte_carlo,
        ode_nstep_max=architecture["ode_nstep_max"],
        ode_method=architecture["ode_method"],
        divergence_method=architecture["divergence_method"],
        ode_kwargs=ode_kwargs,
    )


def build_data_batcher(
    distribution: dict[str, Any],
    shape: int | tuple[int, ...],
    deps: dict[str, Any] | None = None,
) -> Any:
    deps = deps or import_runtime_dependencies()
    name = _distribution_name(distribution)
    if _is_image_distribution(name):
        dataset = load_image_dataset(name)
        return deps["DatasetBatcher"](
            shape,
            distribution.get("resample_each", 1),
            deps["jnp"].asarray(dataset["train"]),
        )

    data_batchers = deps["data_batchers"]
    if name not in data_batchers:
        supported = ", ".join(sorted(data_batchers))
        raise ValueError(
            f"Unsupported data distribution {name!r}; expected one of: {supported}"
        )
    return data_batchers[name](shape, distribution.get("resample_each", 1))


def build_gaussian_logpdf(
    dim: int, distribution: dict[str, Any], deps: dict[str, Any]
) -> Any:
    jnp = deps["jnp"]
    mean_value = distribution.get("mean_value", 0.0)
    sigma_diag = distribution.get("sigma_diag")
    if sigma_diag is None:
        sigma_diag_arr = jnp.ones((dim,))
    else:
        sigma_diag_arr = jnp.asarray(sigma_diag)
        if sigma_diag_arr.ndim == 0:
            sigma_diag_arr = jnp.full((dim,), sigma_diag_arr)
    mean = jnp.full((dim,), mean_value)
    log_norm = 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * sigma_diag_arr))

    def logpdf(x):
        diff = x - mean
        return -0.5 * jnp.sum(diff * diff / sigma_diag_arr, axis=-1) - log_norm

    return logpdf


def build_vectorized_problem(
    problem: dict[str, Any],
    resolved_problem: dict[str, Any],
    method_kwargs: dict[str, Any],
    lane_count: int,
    chunk_rngs: Any,
    template_model: Any,
    deps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deps = deps or import_runtime_dependencies()
    kind = problem["functional"]["kind"]
    distribution = problem["distribution"]
    batch_size = method_kwargs["batch_size"]
    batch_shape = (lane_count, batch_size)

    if kind == "CrossEntropy":
        data_batcher = build_data_batcher(distribution, batch_shape, deps=deps)
        state = {
            "loss": deps["cross_entropy"],
            "next_batch": lambda rngs: data_batcher(rngs),
        }
        name = _distribution_name(distribution)
        if _is_image_distribution(name):
            dataset = load_image_dataset(name)
            state["image_eval_context"] = {
                "distribution_name": name,
                "test": deps["jnp"].asarray(dataset["test"]),
                "mean": deps["jnp"].asarray(dataset["mean"]),
                "std": deps["jnp"].asarray(dataset["std"]),
                "image_shape": dataset["image_shape"],
                "pixel_log_det_per_example": dataset["pixel_log_det_per_example"],
            }
        return state

    if kind == "MMD":
        latent_batcher = deps["LatentBatcherFromModel"](
            batch_shape,
            distribution.get("resample_each", 1),
            template_model,
        )
        data_batcher = build_data_batcher(distribution, batch_shape, deps=deps)
        zipped_batcher = deps["ZipBatcher"](latent_batcher, data_batcher)
        _, bandwidth_batch = zipped_batcher(chunk_rngs)
        bw_multipliers = problem["functional"].get("bw_multipliers")
        if bw_multipliers is not None:
            bw_multipliers = deps["jnp"].asarray(bw_multipliers)
        loss_fn = deps["getMMD"](data=bandwidth_batch[0], bw_multipliers=bw_multipliers)

        def loss(model, batch, rngs):
            del rngs
            return loss_fn(model, batch)

        return {
            "loss": loss,
            "next_batch": lambda rngs: zipped_batcher(rngs),
        }

    if kind == "KL":
        name = _distribution_name(distribution)
        if name == "st":
            logpdf = deps["logpdf_st"]
        elif name == "gaussian":
            logpdf = build_gaussian_logpdf(resolved_problem["dim"], distribution, deps)
        else:
            raise ValueError(f"Unsupported KL distribution {name!r}")
        latent_batcher = deps["LatentBatcherFromModel"](
            batch_shape,
            distribution.get("resample_each", 1),
            template_model,
        )
        return {
            "loss": deps["getKL"](logpdf),
            "next_batch": lambda rngs: latent_batcher(rngs),
        }

    raise ValueError(f"Unsupported functional kind {kind!r}")


def _plotting_config_from_session(session_dir: Path) -> dict[str, Any]:
    normalized_config_path = session_dir / "normalized_config.json"
    if not normalized_config_path.exists():
        return {}
    normalized_config = _load_json(normalized_config_path)
    plotting = normalized_config.get("plotting", {})
    return plotting if isinstance(plotting, dict) else {}


def image_grid_shape(plotting: dict[str, Any]) -> tuple[int, int]:
    nrows = plotting.get("nrows", 3)
    ncols = plotting.get("ncols", 3)
    if not isinstance(nrows, int) or nrows < 1:
        raise ValueError("plotting.nrows must be a positive integer when provided")
    if not isinstance(ncols, int) or ncols < 1:
        raise ValueError("plotting.ncols must be a positive integer when provided")
    return nrows, ncols


def is_image_problem(problem: dict[str, Any]) -> bool:
    return _is_image_distribution(_distribution_name(problem["distribution"]))


def unnormalize_image_samples(samples: Any, image_eval_context: dict[str, Any], deps: dict[str, Any]) -> Any:
    jnp = deps["jnp"]
    mean = image_eval_context["mean"]
    std = image_eval_context["std"]
    image_shape = image_eval_context["image_shape"]
    images = samples.reshape((-1, *image_shape)) * std[None, ...] + mean[None, ...]
    return jnp.clip(images, 0.0, 1.0)


def evaluate_image_model(
    model: Any,
    image_eval_context: dict[str, Any],
    eval_batch_size: int,
    rng_seed: int,
    deps: dict[str, Any],
) -> dict[str, float]:
    import numpy as np

    test_data = image_eval_context["test"]
    pixel_log_det = image_eval_context["pixel_log_det_per_example"]
    total_logp = 0.0
    total_count = 0
    for start in range(0, int(test_data.shape[0]), eval_batch_size):
        batch = test_data[start : start + eval_batch_size]
        _, logp_normalized = model.pullback(
            batch,
            deps["nnx"].Rngs(rng_seed + start),
            with_log_density=True,
        )
        logp_normalized = np.asarray(logp_normalized, dtype=float)
        total_logp += float(logp_normalized.sum()) - pixel_log_det * logp_normalized.shape[0]
        total_count += int(logp_normalized.shape[0])

    nll = -total_logp / total_count
    n_dim = int(np.prod(image_eval_context["image_shape"], dtype=int))
    bits_dim = nll / (n_dim * np.log(2.0))
    return {
        "test_nll": float(nll),
        "test_bits_dim": float(bits_dim),
    }


def build_lane_arrays(
    stacked_metrics: dict[str, Any],
    eval_history: dict[str, list[Any]],
    lane_index: int,
) -> dict[str, Any]:
    import numpy as np

    arrays = {
        name: values[:, lane_index]
        for name, values in stacked_metrics.items()
        if getattr(values, "ndim", 0) >= 2
    }
    for name, values in eval_history.items():
        if not values:
            continue
        lane_values = [value[lane_index] for value in values]
        arrays[name] = np.asarray(lane_values)
    return arrays


METHOD_EXECUTION_KEYS = {
    "max_iterations",
    "batch_size",
    "N_monte_carlo",
    "master_seed",
    "eval_every",
}


def method_factory_param_names(
    method_name: str, deps: dict[str, Any] | None = None
) -> set[str]:
    deps = deps or import_runtime_dependencies()
    method_registry = deps["method_registry"]
    if method_name not in method_registry:
        supported = ", ".join(sorted(method_registry))
        raise ValueError(
            f"Unsupported method {method_name!r}; expected one of: {supported}"
        )

    signature = inspect.signature(method_registry[method_name])
    excluded = {"loss"}
    if method_name in deps["optax_methods"]:
        excluded.add("method")
    return {
        name
        for name, parameter in signature.parameters.items()
        if name not in excluded
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def split_method_kwargs(
    method_name: str, method_kwargs: dict[str, Any], deps: dict[str, Any] | None = None
) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    deps = deps or import_runtime_dependencies()
    kwargs = {
        key: copy.deepcopy(value)
        for key, value in method_kwargs.items()
        if key not in METHOD_EXECUTION_KEYS
    }
    factory_param_names = method_factory_param_names(method_name, deps=deps)
    factory_kwargs = {
        key: kwargs.pop(key) for key in list(kwargs) if key in factory_param_names
    }

    if method_name == "anderson":
        step_size = kwargs.pop("step_size", None)
        relaxation = kwargs.pop("relaxation", 1.0)
        regularization_factor = kwargs.pop(
            "regularization_factor", kwargs.pop("reg_factor", None)
        )
        if step_size is None or regularization_factor is None:
            raise ValueError(
                "Anderson methods require step_size and regularization_factor"
            )
        return factory_kwargs, (step_size, relaxation, regularization_factor), kwargs

    if method_name == "ngd":
        step_size = kwargs.pop("step_size", None)
        if step_size is None:
            raise ValueError("NGD methods require step_size")
        return factory_kwargs, (step_size,), kwargs

    if method_name in deps["optax_methods"]:
        learning_rate = kwargs.pop("learning_rate", kwargs.pop("step_size", None))
        if learning_rate is None:
            raise ValueError(
                f"{method_name} methods require learning_rate or step_size"
            )
        return factory_kwargs, (learning_rate,), kwargs

    raise ValueError(f"Unsupported method {method_name!r}")


def build_method(
    method_name: str,
    factory_kwargs: dict[str, Any],
    loss: Any,
    deps: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    deps = deps or import_runtime_dependencies()
    method_registry = deps["method_registry"]
    if method_name not in method_registry:
        supported = ", ".join(sorted(method_registry))
        raise ValueError(
            f"Unsupported method {method_name!r}; expected one of: {supported}"
        )

    if method_name in deps["optax_methods"]:
        return method_registry[method_name](loss, method=method_name, **factory_kwargs)
    return method_registry[method_name](loss, **factory_kwargs)


def vectorized_method_inputs(
    chunk_runs: list[dict[str, Any]], deps: dict[str, Any]
) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    jnp = deps["jnp"]
    factory_kwargs: dict[str, Any] | None = None
    lane_args: list[tuple[Any, ...]] = []
    lane_kwargs: list[dict[str, Any]] = []

    for planned_run in chunk_runs:
        run_factory_kwargs, run_args, run_kwargs = split_method_kwargs(
            planned_run["method"], planned_run["method_kwargs"], deps=deps
        )
        if factory_kwargs is None:
            factory_kwargs = run_factory_kwargs
        elif factory_kwargs != run_factory_kwargs:
            raise ValueError(
                f"Chunk for method {planned_run['method']} mixes incompatible factory kwargs"
            )
        lane_args.append(run_args)
        lane_kwargs.append(run_kwargs)

    factory_kwargs = factory_kwargs or {}
    if lane_args and lane_args[0]:
        vectorized_args = tuple(
            jnp.asarray([args[index] for args in lane_args])
            for index in range(len(lane_args[0]))
        )
    else:
        vectorized_args = ()

    if lane_kwargs and lane_kwargs[0]:
        keys = list(lane_kwargs[0])
        vectorized_kwargs = {
            key: jnp.asarray([kwargs[key] for kwargs in lane_kwargs]) for key in keys
        }
    else:
        vectorized_kwargs = {}

    return factory_kwargs, vectorized_args, vectorized_kwargs


def build_model_ensemble(
    architecture: dict[str, Any],
    method_kwargs: dict[str, Any],
    dim: int,
    model_rngs: Any,
    deps: dict[str, Any],
) -> Any:
    nnx = deps["nnx"]
    jax = deps["jax"]
    jnp = deps["jnp"]

    ensemble = nnx.vmap(
        lambda rngs: build_model(architecture, method_kwargs, dim, rngs=rngs, deps=deps)
    )(model_rngs)
    graphdef, params, rest = nnx.split(ensemble, nnx.Param, ...)
    params = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf[:1, ...], leaf.shape), params
    )
    return nnx.merge(graphdef, params, rest)


def lane_model_from_ensemble(
    model_ensemble: Any, template_model: Any, lane_index: int, deps: dict[str, Any]
) -> Any:
    nnx = deps["nnx"]
    jax = deps["jax"]

    template_graphdef, _, template_rest = nnx.split(template_model, nnx.Param, ...)
    _, ensemble_params, _ = nnx.split(model_ensemble, nnx.Param, ...)
    lane_params = jax.tree.map(lambda leaf: leaf[lane_index, ...], ensemble_params)
    return nnx.merge(template_graphdef, lane_params, template_rest)


def lane_metric_arrays(
    stacked_metrics: dict[str, Any], lane_index: int
) -> dict[str, Any]:
    return {
        name: values[:, lane_index]
        for name, values in stacked_metrics.items()
        if getattr(values, "ndim", 0) >= 2
    }


def write_run_intermediate_artifacts(
    planned_run: dict[str, Any],
    run_dir: Path,
    model: Any,
    arrays: dict[str, Any],
    deps: dict[str, Any],
    session_dir: Path,
    plotting: dict[str, Any],
    checkpoint_key: str,
    checkpoint_metric: float | None,
    title: str,
    image_eval_context: dict[str, Any] | None = None,
) -> None:
    import numpy as np

    np.savez(run_dir / "arrays.npz", **arrays)
    plot_run_diagnostics(
        model,
        planned_run["problem"],
        arrays,
        title,
        session_dir / "plots" / f"{planned_run['run_id']}_plots.pdf",
        plotting=plotting,
        n_samples=planned_run["method_kwargs"]["batch_size"],
        image_eval_context=image_eval_context,
        deps=deps,
    )
    write_checkpoint(
        run_dir / "checkpoints" / checkpoint_key,
        model,
        deps,
        planned_run,
        checkpoint_key,
        metric_value=checkpoint_metric,
    )


def execute_run_chunk(
    chunk: dict[str, Any],
    session_dir: Path,
    worker_id: int,
    device_info: dict[str, Any],
    config: dict[str, Any],
    deps: dict[str, Any] | None = None,
    progress_position: int | None = None,
) -> list[dict[str, Any]]:
    deps = deps or import_runtime_dependencies()
    nnx = deps["nnx"]
    jax = deps["jax"]
    jnp = deps["jnp"]
    planned_runs = chunk["planned_runs"]
    plotting = config.get("plotting", {})
    session_plot_dir = session_dir / "plots"
    session_plot_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for planned_run in planned_runs:
        run_dir = session_dir / "runs" / planned_run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        write_expanded_run_config(run_dir, planned_run)
        write_status(
            run_dir,
            "running",
            planned_run,
            worker_id,
            device_info,
            started_at=_utc_now_iso(),
            chunk_index=chunk["chunk_index"],
            restart_index=planned_run["restart_index"],
        )
        run_dirs.append(run_dir)

    try:
        reference_run = planned_runs[0]
        max_iterations = reference_run["method_kwargs"].get("max_iterations")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("method_kwargs.max_iterations must be a positive integer")
        eval_every = reference_run["method_kwargs"].get("eval_every")
        if eval_every is not None and (
            not isinstance(eval_every, int) or eval_every < 1
        ):
            raise ValueError(
                "method_kwargs.eval_every must be a positive integer when provided"
            )

        chunk_seed = config["common_params"]["master_seed"] + chunk["chunk_index"]

        master_rngs = nnx.Rngs(config["common_params"]["master_seed"])
        chunk_rngs = nnx.Rngs(chunk_seed)

        lane_rngs = chunk_rngs.fork(split=len(planned_runs))

        template_model = build_model(
            reference_run["architecture"],
            reference_run["method_kwargs"],
            reference_run["resolved_problem"].get(
                "rhs_dim", reference_run["resolved_problem"]["dim"]
            ),
            rngs=master_rngs,
            deps=deps,
        )
        problem_state = build_vectorized_problem(
            reference_run["problem"],
            reference_run["resolved_problem"],
            reference_run["method_kwargs"],
            len(planned_runs),
            chunk_rngs,
            template_model,
            deps=deps,
        )
        factory_kwargs, vectorized_args, vectorized_kwargs = vectorized_method_inputs(
            planned_runs, deps
        )
        init_fun, step_fun = build_method(
            reference_run["method"], factory_kwargs, problem_state["loss"], deps=deps
        )
        image_eval_context = problem_state.get("image_eval_context")
        image_problem = image_eval_context is not None
        vectorized_init = nnx.vmap(init_fun)
        vectorized_step = nnx.jit(nnx.vmap(step_fun))
        ensemble = build_model_ensemble(
            reference_run["architecture"],
            reference_run["method_kwargs"],
            reference_run["resolved_problem"].get(
                "rhs_dim", reference_run["resolved_problem"]["dim"]
            ),
            lane_rngs,
            deps,
        )

        # Set initial weights equal for all the runs!
        _, template_par, _ = nnx.split(template_model, nnx.Param, ...)
        gd, ensemble_par, rest = nnx.split(ensemble, nnx.Param, ...)
        ensemble_par = jax.tree.map(
            lambda _xs, _x0: jnp.broadcast_to(_x0[jnp.newaxis, ...], _xs.shape),
            ensemble_par,
            template_par,
        )
        ensemble = nnx.merge(gd, ensemble_par, rest)

        init_batch = problem_state["next_batch"](master_rngs)
        state = vectorized_init(
            ensemble,
            vectorized_args,
            vectorized_kwargs,
            init_batch,
            lane_rngs,
        )

        metric_names = ("loss", "grad_norm", "natural_grad_norm")
        metric_lists: dict[str, list[Any]] = {name: [] for name in metric_names}
        eval_history: dict[str, list[Any]] = {
            "eval_iteration": [],
            "test_nll": [],
            "test_bits_dim": [],
        }
        best_losses = [float("inf")] * len(planned_runs)
        best_test_nlls = [float("inf")] * len(planned_runs)
        best_test_bits_dims = [None] * len(planned_runs)
        best_eval_iterations = [None] * len(planned_runs)
        last_test_nlls = [None] * len(planned_runs)
        last_test_bits_dims = [None] * len(planned_runs)
        last_eval_iterations = [None] * len(planned_runs)
        progress_bar = None
        if progress_position is not None:
            from tqdm import tqdm

            desc = (
                f"worker {worker_id} {device_info['device_label']} "
                f"{reference_run['method']} x{len(planned_runs)}"
            )
            progress_bar = tqdm(
                total=max_iterations,
                desc=desc,
                position=progress_position,
                leave=False,
            )

        for iteration in range(max_iterations):
            batch = problem_state["next_batch"](chunk_rngs)
            state, values = vectorized_step(state, batch, lane_rngs)
            for name, value in zip(metric_names, values, strict=False):
                metric_lists[name].append(value)

            import numpy as np

            current_losses = np.asarray(values[0], dtype=float)
            model_ensemble = state[0]
            for lane_index, current_loss in enumerate(current_losses):
                if current_loss < best_losses[lane_index]:
                    best_losses[lane_index] = current_loss
                    if not image_problem:
                        lane_model = lane_model_from_ensemble(
                            model_ensemble, template_model, lane_index, deps
                        )
                        write_checkpoint(
                            run_dirs[lane_index] / "checkpoints" / "best",
                            lane_model,
                            deps,
                            planned_runs[lane_index],
                            "best",
                            metric_value=current_loss,
                        )

            if progress_bar is not None:
                progress_bar.update(1)

            should_flush = eval_every is not None and (iteration + 1) % eval_every == 0
            if should_flush:
                stacked_metrics = {
                    name: np.stack([np.asarray(value) for value in values_list], axis=0)
                    for name, values_list in metric_lists.items()
                    if values_list
                }
                if image_problem:
                    eval_iteration = iteration + 1
                    eval_results = []
                    for lane_index in range(len(planned_runs)):
                        lane_model = lane_model_from_ensemble(
                            model_ensemble, template_model, lane_index, deps
                        )
                        metrics = evaluate_image_model(
                            lane_model,
                            image_eval_context,
                            reference_run["method_kwargs"]["batch_size"],
                            reference_run["method_kwargs"].get("master_seed", 0)
                            + eval_iteration
                            + lane_index,
                            deps,
                        )
                        eval_results.append(metrics)
                        last_test_nlls[lane_index] = metrics["test_nll"]
                        last_test_bits_dims[lane_index] = metrics["test_bits_dim"]
                        last_eval_iterations[lane_index] = eval_iteration
                        if metrics["test_nll"] < best_test_nlls[lane_index]:
                            best_test_nlls[lane_index] = metrics["test_nll"]
                            best_test_bits_dims[lane_index] = metrics["test_bits_dim"]
                            best_eval_iterations[lane_index] = eval_iteration
                            write_checkpoint(
                                run_dirs[lane_index] / "checkpoints" / "best",
                                lane_model,
                                deps,
                                planned_runs[lane_index],
                                "best",
                                metric_value=metrics["test_nll"],
                            )
                    eval_history["eval_iteration"].append(
                        np.asarray([eval_iteration] * len(planned_runs), dtype=int)
                    )
                    eval_history["test_nll"].append(
                        np.asarray([item["test_nll"] for item in eval_results], dtype=float)
                    )
                    eval_history["test_bits_dim"].append(
                        np.asarray(
                            [item["test_bits_dim"] for item in eval_results], dtype=float
                        )
                    )
                for lane_index, planned_run in enumerate(planned_runs):
                    lane_arrays = build_lane_arrays(stacked_metrics, eval_history, lane_index)
                    lane_model = lane_model_from_ensemble(
                        model_ensemble, template_model, lane_index, deps
                    )
                    write_run_intermediate_artifacts(
                        planned_run,
                        run_dirs[lane_index],
                        lane_model,
                        lane_arrays,
                        deps,
                        session_dir,
                        plotting,
                        checkpoint_key="last",
                        checkpoint_metric=float(current_losses[lane_index]),
                        title=f"{planned_run['run_id']}, iteration {iteration + 1}",
                        image_eval_context=image_eval_context,
                    )

        if progress_bar is not None:
            progress_bar.close()

        import numpy as np

        stacked_metrics = {
            name: np.stack([np.asarray(value) for value in values_list], axis=0)
            for name, values_list in metric_lists.items()
            if values_list
        }
        final_model_ensemble = state[0]
        if image_problem and last_eval_iterations[0] != max_iterations:
            eval_results = []
            for lane_index in range(len(planned_runs)):
                lane_model = lane_model_from_ensemble(
                    final_model_ensemble, template_model, lane_index, deps
                )
                metrics = evaluate_image_model(
                    lane_model,
                    image_eval_context,
                    reference_run["method_kwargs"]["batch_size"],
                    reference_run["method_kwargs"].get("master_seed", 0)
                    + max_iterations
                    + lane_index,
                    deps,
                )
                eval_results.append(metrics)
                last_test_nlls[lane_index] = metrics["test_nll"]
                last_test_bits_dims[lane_index] = metrics["test_bits_dim"]
                last_eval_iterations[lane_index] = max_iterations
                if metrics["test_nll"] < best_test_nlls[lane_index]:
                    best_test_nlls[lane_index] = metrics["test_nll"]
                    best_test_bits_dims[lane_index] = metrics["test_bits_dim"]
                    best_eval_iterations[lane_index] = max_iterations
                    write_checkpoint(
                        run_dirs[lane_index] / "checkpoints" / "best",
                        lane_model,
                        deps,
                        planned_runs[lane_index],
                        "best",
                        metric_value=metrics["test_nll"],
                    )
            eval_history["eval_iteration"].append(
                np.asarray([max_iterations] * len(planned_runs), dtype=int)
            )
            eval_history["test_nll"].append(
                np.asarray([item["test_nll"] for item in eval_results], dtype=float)
            )
            eval_history["test_bits_dim"].append(
                np.asarray([item["test_bits_dim"] for item in eval_results], dtype=float)
            )
        results: list[dict[str, Any]] = []
        for lane_index, planned_run in enumerate(planned_runs):
            lane_arrays = build_lane_arrays(stacked_metrics, eval_history, lane_index)
            lane_model = lane_model_from_ensemble(
                final_model_ensemble, template_model, lane_index, deps
            )
            final_loss = None
            if "loss" in lane_arrays and lane_arrays["loss"].size > 0:
                final_loss = float(lane_arrays["loss"][-1])
            write_run_intermediate_artifacts(
                planned_run,
                run_dirs[lane_index],
                lane_model,
                lane_arrays,
                deps,
                session_dir,
                plotting,
                checkpoint_key="last",
                checkpoint_metric=final_loss,
                title=f"{planned_run['run_id']}, final",
                image_eval_context=image_eval_context,
            )
            status_extra = {
                "finished_at": _utc_now_iso(),
                "best_loss": best_losses[lane_index],
                "best_checkpoint_criterion": "minimum_loss",
                "arrays_path": "arrays.npz",
                "chunk_index": chunk["chunk_index"],
                "restart_index": planned_run["restart_index"],
            }
            if image_problem:
                status_extra.update(
                    {
                        "best_checkpoint_criterion": "minimum_test_nll_at_eval",
                        "best_test_nll": best_test_nlls[lane_index],
                        "best_test_bits_dim": best_test_bits_dims[lane_index],
                        "best_eval_iteration": best_eval_iterations[lane_index],
                        "last_test_nll": last_test_nlls[lane_index],
                        "last_test_bits_dim": last_test_bits_dims[lane_index],
                        "last_eval_iteration": last_eval_iterations[lane_index],
                    }
                )
            write_status(
                run_dirs[lane_index],
                "success",
                planned_run,
                worker_id,
                device_info,
                **status_extra,
            )
            summary_parts = [
                f"run {planned_run['run_id']} success",
                f"final_loss={final_loss:.6g}" if final_loss is not None else None,
            ]
            if image_problem and last_test_nlls[lane_index] is not None:
                summary_parts.extend(
                    [
                        f"last_test_nll={last_test_nlls[lane_index]:.6g}",
                        f"best_test_nll={best_test_nlls[lane_index]:.6g}",
                        f"last_test_bits_dim={last_test_bits_dims[lane_index]:.6g}",
                    ]
                )
            print("; ".join(part for part in summary_parts if part), flush=True)
            results.append(
                {
                    "status": "success",
                    "run_id": planned_run["run_id"],
                    "run_dir": run_dirs[lane_index],
                }
            )
        return results
    except Exception as exc:
        traceback_text = traceback.format_exc()
        results = []
        for planned_run, run_dir in zip(planned_runs, run_dirs, strict=True):
            write_error_file(run_dir, planned_run, worker_id, exc, traceback_text)
            write_status(
                run_dir,
                "failed",
                planned_run,
                worker_id,
                device_info,
                finished_at=_utc_now_iso(),
                error=str(exc),
                traceback_path="error.txt",
                chunk_index=chunk["chunk_index"],
                restart_index=planned_run["restart_index"],
            )
            results.append(
                {
                    "status": "failed",
                    "run_id": planned_run["run_id"],
                    "run_dir": run_dir,
                    "error": str(exc),
                }
            )
        return results


def run_sequential(
    selected_runs: list[dict[str, Any]],
    session_dir: Path,
    config: dict[str, Any],
) -> dict[str, int]:
    worker_id = 0
    requested_gpu_id = setup_worker_environment(config["parallel"], worker_id)
    device_info = get_worker_device_info(requested_gpu_id, config.get("config"))
    deps = import_runtime_dependencies()
    chunks = chunk_planned_runs(selected_runs, config["parallel"])
    summary = {"success": 0, "failed": 0, "total": len(selected_runs)}
    from tqdm import tqdm

    with tqdm(
        total=len(selected_runs), desc="Runs", position=0, leave=True
    ) as runs_pbar:
        for chunk in chunks:
            results = execute_run_chunk(
                chunk,
                session_dir,
                worker_id,
                device_info,
                config,
                deps=deps,
                progress_position=worker_id + 1,
            )
            for result in results:
                summary[result["status"]] += 1
                runs_pbar.update(1)
    _write_json(session_dir / "summary.json", summary)
    return summary


def worker_loop(
    worker_id: int,
    session_dir: str,
    config: dict[str, Any],
    task_queue: Any,
    message_queue: Any,
) -> None:
    requested_gpu_id = None
    device_info: dict[str, Any] = {
        "requested_gpu_id": None,
        "device_label": "unknown",
        "jax_devices": [],
        "jax_platforms": [],
    }
    try:
        requested_gpu_id = setup_worker_environment(config["parallel"], worker_id)
        device_info = get_worker_device_info(requested_gpu_id, config.get("config"))
        message_queue.put(
            {
                "event": "worker_started",
                "worker_id": worker_id,
                **device_info,
            }
        )
        deps = import_runtime_dependencies()
        session_path = Path(session_dir)

        while True:
            chunk = task_queue.get()
            if chunk is None:
                message_queue.put(
                    {"event": "worker_empty_queue", "worker_id": worker_id}
                )
                break

            message_queue.put(
                {
                    "event": "chunk_started",
                    "worker_id": worker_id,
                    "chunk_index": chunk["chunk_index"],
                    "method": chunk["method"],
                    "run_ids": [
                        planned_run["run_id"] for planned_run in chunk["planned_runs"]
                    ],
                }
            )
            results = execute_run_chunk(
                chunk,
                session_path,
                worker_id,
                device_info,
                config,
                deps=deps,
                progress_position=worker_id + 1,
            )
            message_queue.put(
                {
                    "event": "chunk_complete",
                    "worker_id": worker_id,
                    "chunk_index": chunk["chunk_index"],
                    "results": [
                        {
                            "status": result["status"],
                            "run_id": result["run_id"],
                            "error": result.get("error"),
                        }
                        for result in results
                    ],
                }
            )
    except Exception as exc:
        message_queue.put(
            {
                "event": "worker_error",
                "worker_id": worker_id,
                "requested_gpu_id": requested_gpu_id,
                **device_info,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        message_queue.put({"event": "worker_exit", "worker_id": worker_id})


def run_parallel(
    selected_runs: list[dict[str, Any]],
    session_dir: Path,
    config: dict[str, Any],
) -> dict[str, int]:
    from tqdm import tqdm

    chunks = chunk_planned_runs(selected_runs, config["parallel"])
    n_workers = min(len(config["parallel"]["gpu_ids"]), len(chunks))
    if n_workers <= 1:
        return run_sequential(selected_runs, session_dir, config)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    message_queue = ctx.Queue()
    for chunk in chunks:
        task_queue.put(chunk)
    for _ in range(n_workers):
        task_queue.put(None)

    processes = []
    for worker_id in range(n_workers):
        process = ctx.Process(
            target=worker_loop,
            args=(worker_id, str(session_dir), config, task_queue, message_queue),
        )
        process.start()
        processes.append(process)

    summary = {"success": 0, "failed": 0, "total": len(selected_runs)}
    exited_workers: set[int] = set()

    with tqdm(
        total=len(selected_runs), desc="Runs", position=0, leave=True
    ) as runs_pbar:
        while len(exited_workers) < n_workers:
            try:
                message = message_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            event = message.get("event")
            worker_id = message.get("worker_id")
            if event == "worker_started":
                tqdm.write(
                    f"worker {worker_id} started on {message.get('device_label')} "
                    f"devices={message.get('jax_devices')}"
                )
            elif event == "chunk_started":
                tqdm.write(
                    f"worker {worker_id} started chunk {message.get('chunk_index')} "
                    f"method={message.get('method')} runs={message.get('run_ids')}"
                )
            elif event == "chunk_complete":
                for result in message.get("results", []):
                    summary[result["status"]] += 1
                    if result["status"] == "failed":
                        tqdm.write(
                            f"worker {worker_id} failed {result['run_id']}: {result.get('error')}"
                        )
                    runs_pbar.update(1)
            elif event == "worker_error":
                tqdm.write(
                    f"worker {worker_id} orchestration error: {message.get('error')}"
                )
            elif event == "worker_empty_queue":
                tqdm.write(f"worker {worker_id} queue empty")
            elif event == "worker_exit":
                exited_workers.add(worker_id)

    for process in processes:
        process.join()
        if process.exitcode not in {0, None}:
            raise RuntimeError(f"parallel worker exited with code {process.exitcode}")

    _write_json(session_dir / "summary.json", summary)
    return summary


def _load_saved_arrays(arrays_path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(arrays_path) as loaded_arrays:
        return {name: loaded_arrays[name] for name in loaded_arrays.files}


def entry_arrays(entry: dict[str, Any]) -> dict[str, Any]:
    return {name: entry[name] for name in entry.get("array_names", []) if name in entry}


def _expanded_config_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    config_keys = (
        "run_index",
        "run_id",
        "restart_index",
        "problem",
        "resolved_problem",
        "architecture",
        "method",
        "method_kwargs",
    )
    return {key: copy.deepcopy(entry[key]) for key in config_keys if key in entry}


ENTRY_STATUS_KEYS = {
    "best_checkpoint_criterion",
    "best_loss",
    "best_test_nll",
    "best_test_bits_dim",
    "best_eval_iteration",
    "last_test_nll",
    "last_test_bits_dim",
    "last_eval_iteration",
}


def load_experiment_entries(session_dir: str | Path) -> list[dict[str, Any]]:
    """Load successful benchmark runs as plain analysis entries.

    Each returned entry contains the expanded config fields at top level,
    unpacked saved arrays, checkpoint paths, and derived loss summary metrics.
    Failed and incomplete runs are skipped.
    """

    session_dir = Path(session_dir)
    runs_dir = session_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Benchmark runs directory does not exist: {runs_dir}")

    entries: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        status_path = run_dir / "status.json"
        expanded_config_path = run_dir / "expanded_config.json"
        arrays_path = run_dir / "arrays.npz"
        if (
            not status_path.exists()
            or not expanded_config_path.exists()
            or not arrays_path.exists()
        ):
            continue

        status = _load_json(status_path)
        if status.get("status") != "success":
            continue

        arrays = _load_saved_arrays(arrays_path)
        expanded_config = _load_json(expanded_config_path)
        entry: dict[str, Any] = copy.deepcopy(expanded_config)
        entry["run_dir"] = run_dir
        entry["checkpoints"] = {
            "best": (run_dir / "checkpoints" / "best").resolve(),
            "last": (run_dir / "checkpoints" / "last").resolve(),
        }
        for key in ENTRY_STATUS_KEYS:
            if key in status:
                entry[key] = copy.deepcopy(status[key])
        entry["array_names"] = list(arrays)
        entry.update(arrays)

        loss = arrays.get("loss")
        if loss is not None and getattr(loss, "size", 0) > 0:
            import numpy as np

            loss_array = np.asarray(loss, dtype=float)
            best_iteration = int(np.argmin(loss_array))
            entry["best_loss"] = float(loss_array[best_iteration])
            entry["best_iteration"] = best_iteration
            entry["final_loss"] = float(loss_array[-1])

        entries.append(entry)

    return entries


def load_completed_runs(session_dir: str | Path) -> list[dict[str, Any]]:
    """Compatibility wrapper over :func:`load_experiment_entries`."""

    entries = load_experiment_entries(session_dir)
    completed_runs: list[dict[str, Any]] = []
    for entry in entries:
        completed_runs.append(
            {
                "run_id": entry.get("run_id"),
                "run_dir": entry["run_dir"],
                "status": {"status": "success"},
                "expanded_config": _expanded_config_from_entry(entry),
                "arrays": entry_arrays(entry),
            }
        )
    return completed_runs


def _value_identity(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _flatten_mapping(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_mapping(child_prefix, child, output)
    else:
        output[prefix] = value


def flatten_run_params(expanded_config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if "method" in expanded_config:
        flat["method"] = expanded_config["method"]
    if "restart_index" in expanded_config:
        flat["restart_index"] = expanded_config["restart_index"]
    if "method_kwargs" in expanded_config:
        _flatten_mapping("method_kwargs", expanded_config["method_kwargs"], flat)
    if "architecture" in expanded_config:
        _flatten_mapping("architecture", expanded_config["architecture"], flat)
    if "problem" in expanded_config:
        _flatten_mapping("problem", expanded_config["problem"], flat)
    return flat


def flatten_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return flatten_run_params(entry)


def varying_param_keys(expanded_configs: list[dict[str, Any]]) -> list[str]:
    flattened = [flatten_entry(config) for config in expanded_configs]
    keys: list[str] = []
    for flat in flattened:
        for key in flat:
            if key not in keys:
                keys.append(key)

    varying: list[str] = []
    for key in keys:
        values = {_value_identity(flat[key]) for flat in flattened if key in flat}
        if len(values) > 1:
            varying.append(key)
    return varying


def _display_param_key(key: str) -> str:
    for prefix in ("method_kwargs.", "architecture.", "problem."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def format_run_label(expanded_config: dict[str, Any], keys: list[str]) -> str:
    flat = flatten_entry(expanded_config)
    parts = [f"{_display_param_key(key)} {flat[key]}" for key in keys if key in flat]
    return ", ".join(parts) if parts else expanded_config.get("run_id", "run")


def _loss_array(item: dict[str, Any]) -> Any:
    if "loss" in item:
        return item["loss"]
    raise KeyError("get_lines input item is missing a loss array")


def _ordered_keys(flattened: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for flat in flattened:
        for key in flat:
            if key not in keys:
                keys.append(key)
    return keys


def _varying_keys_from_flattened(flattened: list[dict[str, Any]]) -> list[str]:
    varying: list[str] = []
    for key in _ordered_keys(flattened):
        values = {_value_identity(flat[key]) for flat in flattened if key in flat}
        if len(values) > 1:
            varying.append(key)
    return varying


def _unique_values(flattened: list[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    identities: set[str] = set()
    for flat in flattened:
        if key not in flat:
            continue
        identity = _value_identity(flat[key])
        if identity not in identities:
            identities.add(identity)
            values.append(flat[key])
    return values


def _normalize_style_channel_name(name: str) -> str:
    return "colormap" if name == "cmap" else name


def _combo_identity(flat: dict[str, Any], keys: list[str]) -> str:
    return json.dumps([flat.get(key) for key in keys], sort_keys=True, default=str)


def _resolve_style_channel_keys(
    varying_keys: list[str], style_channel_map: dict[str, Any]
) -> dict[str, str]:
    display_to_keys: dict[str, list[str]] = {}
    for key in varying_keys:
        display_to_keys.setdefault(_display_param_key(key), []).append(key)

    resolved: dict[str, str] = {}
    for requested_key, channel_name in style_channel_map.items():
        if requested_key in varying_keys:
            resolved[requested_key] = _normalize_style_channel_name(str(channel_name))
            continue
        display_matches = display_to_keys.get(requested_key, [])
        if len(display_matches) == 1:
            resolved[display_matches[0]] = _normalize_style_channel_name(
                str(channel_name)
            )
    return resolved


def _group_items_by_keys(
    items: list[dict[str, Any]], keys: list[str]
) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    order: list[tuple[str, ...]] = []
    for item in items:
        flat = flatten_entry(item)
        group_key = tuple(_value_identity(flat.get(key)) for key in keys)
        if group_key not in groups:
            groups[group_key] = []
            order.append(group_key)
        groups[group_key].append(item)
    return [(key, groups[key]) for key in order]


def architecture_group_fields(entry: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_entry(entry)
    return {
        key: value for key, value in flat.items() if key.startswith("architecture.")
    }


def method_group_fields(entry: dict[str, Any]) -> dict[str, Any]:
    fields = {"method": entry.get("method")}
    for key, value in split_method_kwargs(entry["method"], entry["method_kwargs"])[
        0
    ].items():
        fields[f"method_kwargs.{key}"] = value
    return fields


def restart_group_fields(entry: dict[str, Any]) -> dict[str, Any]:
    flat = flatten_entry(entry)
    return {key: value for key, value in flat.items() if key != "restart_index"}


def _frame_flattened_columns(frame: Any) -> list[str]:
    return [
        key
        for key in frame.columns
        if key == "method"
        or key == "restart_index"
        or key.startswith(("method_kwargs.", "architecture.", "problem."))
    ]


def architecture_group_columns(frame: Any) -> list[str]:
    return [
        key
        for key in _frame_flattened_columns(frame)
        if key.startswith("architecture.")
    ]


def method_group_columns(frame: Any) -> list[str]:
    columns = ["method"]
    method_series = frame["method"] if "method" in frame.columns else None
    method_names = (
        [] if method_series is None else method_series.dropna().unique().tolist()
    )
    for method_name in method_names:
        for key in sorted(method_factory_param_names(method_name)):
            column = f"method_kwargs.{key}"
            if column in frame.columns and column not in columns:
                columns.append(column)
    return columns


def restart_group_columns(frame: Any) -> list[str]:
    return [key for key in _frame_flattened_columns(frame) if key != "restart_index"]


def entries_to_frame(entries: list[dict[str, Any]]) -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(entries):
        row = {
            "entry_index": entry_index,
            "run_id": entry.get("run_id"),
            "run_dir": entry.get("run_dir"),
            "checkpoint_best": entry.get("checkpoints", {}).get("best"),
            "checkpoint_last": entry.get("checkpoints", {}).get("last"),
            "best_loss": entry.get("best_loss"),
            "best_iteration": entry.get("best_iteration"),
            "final_loss": entry.get("final_loss"),
        }
        row.update(flatten_entry(entry))
        rows.append(row)
    return pd.DataFrame(rows)


def entries_in_same_restart_group(
    entries: list[dict[str, Any]], entry: dict[str, Any]
) -> list[dict[str, Any]]:
    if not entries:
        return []

    frame = entries_to_frame(entries)
    target_fields = restart_group_fields(entry)
    mask = None
    for key in restart_group_columns(frame):
        value = target_fields.get(key)
        current = frame[key].isna() if value is None else frame[key] == value
        mask = current if mask is None else (mask & current)
    if mask is None:
        return entries
    indices = frame.loc[mask, "entry_index"].tolist()
    return [entries[index] for index in indices]


def entries_in_same_restart_group_at_index(
    entries: list[dict[str, Any]], index: int
) -> list[dict[str, Any]]:
    return entries_in_same_restart_group(entries, entries[index])


def _filename_suffix(index: int, group_key: dict[str, Any]) -> str:
    if not group_key:
        return ""
    return f"arch_{index:03d}"


def _representative_params(
    items: list[dict[str, Any]], keys: list[str]
) -> dict[str, Any]:
    base = dict(items[0])
    flat = flatten_entry(base)
    base["_line_params"] = {key: flat[key] for key in keys if key in flat}
    return base


def _line_label(representative: dict[str, Any], keys: list[str]) -> str:
    line_params = representative.get("_line_params")
    if isinstance(line_params, dict):
        parts = [
            f"{_display_param_key(key)} {line_params[key]}"
            for key in keys
            if key in line_params
        ]
        if parts:
            return ", ".join(parts)
    return format_run_label(representative, keys)


def _assign_line_styles(
    representatives: list[dict[str, Any]],
    style_channels: dict[str, Any],
    style_channel_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not representatives:
        return []

    flattened = []
    for representative in representatives:
        line_params = representative.get("_line_params")
        if isinstance(line_params, dict):
            flattened.append(line_params)
        else:
            flattened.append(flatten_entry(representative))

    varying_keys = _varying_keys_from_flattened(flattened)
    if style_channel_map:
        import matplotlib.pyplot as plt

        resolved_map = _resolve_style_channel_keys(varying_keys, style_channel_map)
        if not resolved_map:
            return _assign_line_styles(representatives, style_channels, None)

        channel_to_keys: dict[str, list[str]] = {}
        for key in varying_keys:
            channel_name = resolved_map.get(key)
            if channel_name is not None:
                channel_to_keys.setdefault(channel_name, []).append(key)

        ordered_channels = list(channel_to_keys)
        if ordered_channels:
            fallback_channel = ordered_channels[-1]
            for key in varying_keys:
                if key not in resolved_map:
                    channel_to_keys.setdefault(fallback_channel, []).append(key)

        colormaps = style_channels.get("colormap", []) if style_channels else []
        list_channels = {
            key: value
            for key, value in (style_channels or {}).items()
            if key != "colormap" and isinstance(value, list) and value
        }
        colormap_keys = channel_to_keys.get("colormap", [])
        color_keys = channel_to_keys.get("color", [])

        cmap_by_identity: dict[str, str] = {}
        if colormap_keys and colormaps:
            combo_order: list[str] = []
            for flat in flattened:
                combo_id = _combo_identity(flat, colormap_keys)
                if combo_id not in cmap_by_identity:
                    combo_order.append(combo_id)
                    cmap_by_identity[combo_id] = colormaps[
                        (len(combo_order) - 1) % len(colormaps)
                    ]

        styles: list[dict[str, Any]] = []
        for flat in flattened:
            style: dict[str, Any] = {}

            colormap_identity = (
                _combo_identity(flat, colormap_keys) if colormap_keys else "__default__"
            )
            cmap_name = cmap_by_identity.get(
                colormap_identity,
                colormaps[0] if colormaps else "viridis",
            )

            if color_keys:
                shade_identity = _combo_identity(flat, color_keys)
                shade_values: list[str] = []
                for other_flat in flattened:
                    if colormap_keys and (
                        _combo_identity(other_flat, colormap_keys) != colormap_identity
                    ):
                        continue
                    other_identity = _combo_identity(other_flat, color_keys)
                    if other_identity not in shade_values:
                        shade_values.append(other_identity)
                shade_index = shade_values.index(shade_identity)
                shade = (shade_index + 1) / (len(shade_values) + 1)
                style["color"] = plt.get_cmap(cmap_name)(shade)
            elif colormap_keys and colormaps:
                style["color"] = plt.get_cmap(cmap_name)(0.6)

            for channel_name, channel_values in list_channels.items():
                channel_keys = channel_to_keys.get(channel_name, [])
                if not channel_keys:
                    continue
                combo_id = _combo_identity(flat, channel_keys)
                combo_order: list[str] = []
                for other_flat in flattened:
                    other_id = _combo_identity(other_flat, channel_keys)
                    if other_id not in combo_order:
                        combo_order.append(other_id)
                value_index = combo_order.index(combo_id)
                style[channel_name] = channel_values[value_index % len(channel_values)]

            styles.append(style)

        return styles

    colormaps = style_channels.get("colormap", []) if style_channels else []
    list_channels = [
        (key, value)
        for key, value in (style_channels or {}).items()
        if key != "colormap" and isinstance(value, list) and value
    ]

    cmap_by_identity: dict[str, str] = {}
    color_key = varying_keys[0] if varying_keys else None
    shade_key = varying_keys[1] if len(varying_keys) > 1 else None
    if color_key is not None and colormaps:
        for index, value in enumerate(_unique_values(flattened, color_key)):
            cmap_by_identity[_value_identity(value)] = colormaps[index % len(colormaps)]

    styles: list[dict[str, Any]] = []
    for flat in flattened:
        style: dict[str, Any] = {}
        if color_key is not None and colormaps:
            import matplotlib.pyplot as plt

            cmap_name = cmap_by_identity[_value_identity(flat[color_key])]
            if shade_key is not None and shade_key in flat:
                shade_values = _unique_values(flattened, shade_key)
                shade_index = next(
                    index
                    for index, value in enumerate(shade_values)
                    if _value_identity(value) == _value_identity(flat[shade_key])
                )
                shade = (shade_index + 1) / (len(shade_values) + 1)
            else:
                shade = 0.6
            style["color"] = plt.get_cmap(cmap_name)(shade)

        for channel_index, param_key in enumerate(varying_keys[2:]):
            if channel_index >= len(list_channels) or param_key not in flat:
                continue
            channel_name, channel_values = list_channels[channel_index]
            unique_values = _unique_values(flattened, param_key)
            value_index = next(
                index
                for index, value in enumerate(unique_values)
                if _value_identity(value) == _value_identity(flat[param_key])
            )
            style[channel_name] = channel_values[value_index % len(channel_values)]

        styles.append(style)

    return styles


def get_lines(
    entries: list[dict[str, Any]],
    style_channels: dict[str, Any] | None = None,
    style_channel_map: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build aggregate plotting groups and loss series definitions.

    Architecture-varying parameters define top-level plot groups. Runs that
    differ only by ``restart_index`` are aggregated into mean +/- std loss
    tubes within each architecture group.
    """

    if not entries:
        return []

    import numpy as np

    frame = entries_to_frame(entries)
    flattened_all = [flatten_entry(entry) for entry in entries]
    varying_keys_all = _varying_keys_from_flattened(flattened_all)
    architecture_keys = [
        key for key in architecture_group_columns(frame) if key in varying_keys_all
    ]
    if architecture_keys:
        architecture_groups = [
            group
            for _, group in frame.groupby(architecture_keys, sort=False, dropna=False)
        ]
    else:
        architecture_groups = [frame]
    groups: list[dict[str, Any]] = []

    for group_index, group_frame in enumerate(architecture_groups):
        group_items = [entries[index] for index in group_frame["entry_index"].tolist()]
        group_flat = [flatten_entry(item) for item in group_items]
        architecture_group_key = {
            key: group_flat[0][key] for key in architecture_keys if key in group_flat[0]
        }
        title = (
            ", ".join(
                f"{_display_param_key(key)} {value}"
                for key, value in architecture_group_key.items()
            )
            if architecture_group_key
            else "aggregate loss"
        )

        varying_keys = _varying_keys_from_flattened(group_flat)
        line_param_keys = [
            key
            for key in restart_group_columns(group_frame)
            if key in varying_keys and key not in architecture_keys
        ]

        line_items: list[dict[str, Any]] = []
        representatives: list[dict[str, Any]] = []
        if line_param_keys:
            restart_groups = [
                group
                for _, group in group_frame.groupby(
                    line_param_keys, sort=False, dropna=False
                )
            ]
        else:
            restart_groups = [group_frame]

        for restart_group in restart_groups:
            restart_group_items = [
                entries[index] for index in restart_group["entry_index"].tolist()
            ]
            representative = _representative_params(
                restart_group_items, line_param_keys
            )
            representatives.append(representative)
            losses = [
                np.asarray(_loss_array(item), dtype=float)
                for item in restart_group_items
            ]
            if len(losses) > 1:
                min_length = min(loss.shape[0] for loss in losses)
                if any(loss.shape[0] != min_length for loss in losses):
                    warnings.warn(
                        "Loss arrays in a restart group have different lengths; truncating to minimum length.",
                        stacklevel=2,
                    )
                stacked = np.stack([loss[:min_length] for loss in losses], axis=0)
                mean = stacked.mean(axis=0)
                std = stacked.std(axis=0)
                line_items.append(
                    {
                        "label": _line_label(representative, line_param_keys),
                        "loss_mean": mean,
                        "loss_upper": mean + std,
                        "loss_lower": mean - std,
                    }
                )
            else:
                item = restart_group_items[0]
                label = _line_label(representative, line_param_keys) or item.get(
                    "run_id", "run"
                )
                line_items.append(
                    {
                        "label": label,
                        "loss": np.asarray(_loss_array(item), dtype=float),
                    }
                )

        styles = _assign_line_styles(
            representatives,
            style_channels or {},
            style_channel_map=style_channel_map,
        )
        for line, style in zip(line_items, styles, strict=True):
            line["style"] = style

        groups.append(
            {
                "group_key": architecture_group_key,
                "title": title,
                "filename_suffix": _filename_suffix(
                    group_index, architecture_group_key
                ),
                "lines": line_items,
            }
        )

    return groups


def _plot_aggregate_group(group: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for line in group["lines"]:
        style = line.get("style", {})
        color = style.get("color")
        linestyle = style.get("linestyle", "-")
        linewidth = style.get("linewidth", 1.5)

        if "loss_mean" in line:
            iterations = np.arange(line["loss_mean"].shape[0])
            ax.plot(
                iterations,
                line["loss_mean"],
                label=line["label"],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            if "loss_upper" in line and "loss_lower" in line:
                ax.fill_between(
                    iterations,
                    line["loss_lower"],
                    line["loss_upper"],
                    color=color,
                    alpha=0.2,
                )
        elif "loss" in line:
            loss = np.asarray(line["loss"], dtype=float)
            iterations = np.arange(loss.shape[0])
            ax.plot(
                iterations,
                loss,
                label=line["label"],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(group["title"])
    ax.legend()
    ax.grid()
    fig.savefig(output_path)
    plt.close(fig)


def generate_aggregate_plots(
    session_dir: str | Path,
    plotting: dict[str, Any] | None = None,
) -> list[Path]:
    session_dir = Path(session_dir)
    plotting = plotting or _plotting_config_from_session(session_dir)
    entries = load_experiment_entries(session_dir)
    if not entries:
        return []

    style_channels = plotting.get("style_channels", {})
    style_channel_map = plotting.get("style_channel_map")
    groups = get_lines(entries, style_channels, style_channel_map=style_channel_map)
    if not groups:
        return []

    plot_dir = session_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for group in groups:
        suffix = group["filename_suffix"]
        filename = f"aggregate_loss_{suffix}.pdf" if suffix else "aggregate_loss.pdf"
        output_path = plot_dir / filename
        _plot_aggregate_group(group, output_path)
        written_paths.append(output_path)

    return written_paths


def plot_run_diagnostics(
    model: Any,
    problem: dict[str, Any],
    arrays: dict[str, Any],
    title: str,
    output_path: Path,
    plotting: dict[str, Any] | None = None,
    n_samples: int = 512,
    image_eval_context: dict[str, Any] | None = None,
    deps: dict[str, Any] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plotting = plotting or {}
    deps = deps or import_runtime_dependencies()
    image_problem = is_image_problem(problem)
    if image_problem:
        nrows, ncols = image_grid_shape(plotting)
        n_samples = nrows * ncols

    x = model.sample(n_samples, deps["nnx"].Rngs(0))
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 8), layout="constrained")

    ax = axs[0]
    if image_problem:
        if image_eval_context is None:
            dataset = load_image_dataset(_distribution_name(problem["distribution"]))
            image_eval_context = {
                "mean": deps["jnp"].asarray(dataset["mean"]),
                "std": deps["jnp"].asarray(dataset["std"]),
                "image_shape": dataset["image_shape"],
            }
        images = unnormalize_image_samples(x, image_eval_context, deps)
        images = images.reshape((nrows, ncols, *image_eval_context["image_shape"]))
        canvas = images.transpose((0, 2, 1, 3)).reshape(
            nrows * image_eval_context["image_shape"][0],
            ncols * image_eval_context["image_shape"][1],
        )
        ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title("Generated samples")
        ax.axis("off")
    elif x.shape[-1] >= 2:
        ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}(z)$", marker="*", s=5.0)
        ax.legend()
    else:
        ax.hist(x[:, 0], bins=40, label=r"$T_{\text{opt}}(z)$")
        ax.legend()

    ax = axs[1]
    if "loss" in arrays:
        ax.plot(arrays["loss"], label="Loss")
    if "eval_iteration" in arrays and "test_nll" in arrays:
        ax.plot(arrays["eval_iteration"], arrays["test_nll"], label="Test NLL")
    ax.legend()

    ax = axs[2]
    if "grad_norm" in arrays:
        ax.plot(arrays["grad_norm"], color="red", label=r"$\| \nabla L\|_2$")
        ax.set_yscale("log")
    if "natural_grad_norm" in arrays:
        ax1 = ax.twinx()
        ax1.plot(
            arrays["natural_grad_norm"],
            color="tab:orange",
            label=r"$\| \partial_W L\|_2$",
        )
        ax1.set_yscale("log")
    ax.legend()

    for ax in axs:
        if ax is not axs[0] or not image_problem:
            ax.grid()
    fig.suptitle(title)
    fig.savefig(output_path)
    plt.close(fig)


def generate_per_run_plots(
    session_dir: str | Path,
    plotting: dict[str, Any] | None = None,
) -> list[Path]:
    session_dir = Path(session_dir)
    plotting = plotting or _plotting_config_from_session(session_dir)
    entries = load_experiment_entries(session_dir)
    if not entries:
        return []

    plot_dir = session_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    planned_runs_path = session_dir / "planned_runs.json"
    if planned_runs_path.exists():
        label_entries = _load_json_value(planned_runs_path)
    else:
        label_entries = entries
    varying_keys = varying_param_keys(label_entries)
    n_samples = plotting.get("n_samples_plot", plotting.get("n_samples", 512))
    written_paths: list[Path] = []

    for entry in entries:
        model = load_entry_model(entry, key="last")
        title = format_run_label(entry, varying_keys)
        output_path = plot_dir / f"{entry['run_id']}_plots.pdf"
        plot_run_diagnostics(
            model,
            entry["problem"],
            entry_arrays(entry),
            title,
            output_path,
            plotting=plotting,
            n_samples=n_samples,
        )
        written_paths.append(output_path)

    return written_paths


def load_entry_model(entry: dict[str, Any], key: str = "last") -> Any:
    if key not in {"last", "best"}:
        raise ValueError("checkpoint key must be either 'last' or 'best'")

    checkpoint_dir = Path(entry["checkpoints"][key]).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    deps = import_runtime_dependencies()
    model = build_model(
        entry["architecture"],
        entry["method_kwargs"],
        entry["resolved_problem"].get("rhs_dim", entry["resolved_problem"]["dim"]),
        rngs=deps["nnx"].Rngs(entry["method_kwargs"].get("master_seed", 0)),
        deps=deps,
    )

    import orbax.checkpoint as ocp

    graphdef, state = deps["nnx"].split(model)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(checkpoint_dir, target=state)
    return deps["nnx"].merge(graphdef, state)


def load_model_checkpoint(
    session_dir: str | Path,
    run_id: str,
    key: str = "last",
) -> Any:
    entries = load_experiment_entries(session_dir)
    for entry in entries:
        if entry.get("run_id") == run_id:
            return load_entry_model(entry, key=key)
    raise FileNotFoundError(f"Unknown run_id or missing expanded config: {run_id}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, validate, and normalize a benchmark config.

    Stage 3 validation is intentionally schema-level only. Data-dependent
    problem resolution, dimension inference, and run planning are later stages.
    """

    config_path = Path(path)
    config = copy.deepcopy(_load_json(config_path))

    keys = set(config)
    missing = REQUIRED_TOP_LEVEL_KEYS - keys
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required top-level config keys: {names}")

    allowed = REQUIRED_TOP_LEVEL_KEYS | OPTIONAL_TOP_LEVEL_KEYS
    unknown = keys - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown top-level config keys are not supported: {names}")

    if not isinstance(config["output_root"], str):
        raise ValueError("output_root must be a string path")
    common_params = _require_object(config["common_params"], "common_params")
    if "N_samples" in common_params:
        raise ValueError(
            "common_params.N_samples is not supported; use common_params.batch_size"
        )
    if "plot_every" in common_params:
        raise ValueError(
            "common_params.plot_every is not supported; use common_params.eval_every"
        )
    batch_size = common_params.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("common_params.batch_size must be a positive integer")
    eval_every = common_params.get("eval_every")
    if eval_every is not None and (
        not isinstance(eval_every, int) or eval_every < 1
    ):
        raise ValueError("common_params.eval_every must be a positive integer when provided")
    problem = _require_object(config["problem"], "problem")
    architectures = _require_non_empty_object_list(
        config["architectures"], "architectures"
    )
    methods = _require_non_empty_object_list(config["methods"], "methods")
    plotting = _require_object(config["plotting"], "plotting")
    if plotting.get("nrows") is not None and (
        not isinstance(plotting["nrows"], int) or plotting["nrows"] < 1
    ):
        raise ValueError("plotting.nrows must be a positive integer when provided")
    if plotting.get("ncols") is not None and (
        not isinstance(plotting["ncols"], int) or plotting["ncols"] < 1
    ):
        raise ValueError("plotting.ncols must be a positive integer when provided")
    style_channel_map = plotting.get("style_channel_map")
    if style_channel_map is not None:
        if not isinstance(style_channel_map, dict) or not style_channel_map:
            raise ValueError("plotting.style_channel_map must be a non-empty object when provided")
        valid_channels = {"cmap", "colormap", "color"}
        valid_channels.update(
            key
            for key, value in plotting.get("style_channels", {}).items()
            if key != "colormap" and isinstance(value, list) and value
        )
        for key, value in style_channel_map.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "plotting.style_channel_map keys must be non-empty strings"
                )
            if not isinstance(value, str) or value not in valid_channels:
                supported = ", ".join(sorted(valid_channels))
                raise ValueError(
                    f"plotting.style_channel_map[{key!r}] must be one of: {supported}"
                )

    master_seed = common_params.setdefault("master_seed", 0)
    if not isinstance(master_seed, int):
        raise ValueError("common_params.master_seed must be an integer")

    functional = _require_object(problem.get("functional"), "problem.functional")
    if "name" in functional:
        raise ValueError(
            "problem.functional.name is not supported; use functional.kind"
        )
    kind = functional.get("kind")
    if kind not in SUPPORTED_FUNCTIONAL_KINDS:
        supported = ", ".join(sorted(SUPPORTED_FUNCTIONAL_KINDS))
        raise ValueError(f"problem.functional.kind must be exactly one of: {supported}")

    if "distribution" not in problem:
        raise ValueError("problem.distribution is required")
    distribution = problem["distribution"]
    if isinstance(distribution, str):
        problem["distribution"] = {"name": distribution}
    elif not isinstance(distribution, dict):
        raise ValueError("problem.distribution must be either a string or an object")
    elif "n_samples" in distribution:
        raise ValueError(
            "problem.distribution.n_samples is not supported; use common_params.batch_size"
        )

    for index, architecture in enumerate(architectures):
        _validate_architecture(architecture, index)

    for index, method in enumerate(methods):
        _validate_grid_axes(method, f"methods[{index}]")
        method_name = method.get("method")
        if not isinstance(method_name, str):
            raise ValueError(f"methods[{index}].method must be a string")
        if method_name not in str_to_method:
            supported = ", ".join(sorted(str_to_method))
            raise ValueError(
                f"methods[{index}].method {method_name!r} is not supported; "
                f"expected one of: {supported}"
            )
        n_restarts = method.get("n_restarts")
        if not isinstance(n_restarts, int) or n_restarts < 1:
            raise ValueError(
                f"methods[{index}].n_restarts must be an integer greater than or equal to 1"
            )

    parallel = config.setdefault("parallel", {})
    _require_object(parallel, "parallel")
    gpu_ids = parallel.get("gpu_ids")
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ValueError("parallel.gpu_ids must be a non-empty list")
    for index, gpu_id in enumerate(gpu_ids):
        if not isinstance(gpu_id, (int, str)):
            raise ValueError(
                f"parallel.gpu_ids[{index}] must be an integer or string device id"
            )

    max_parallel = _require_object(
        parallel.get("max_parallel"), "parallel.max_parallel"
    )
    for method in methods:
        method_name = method["method"]
        method_parallel = max_parallel.get(method_name)
        if not isinstance(method_parallel, int) or method_parallel < 1:
            raise ValueError(
                f"parallel.max_parallel.{method_name} must be an integer greater than or equal to 1"
            )

    jax_config = config.setdefault("config", [])
    if not isinstance(jax_config, list):
        raise ValueError("config must be a list")
    for index, entry in enumerate(jax_config):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"config[{index}] must be a [flag, value] pair")
        flag, value = entry
        if not isinstance(flag, str):
            raise ValueError(
                f"config[{index}] flag must be a string, got {type(flag).__name__}"
            )

    config["resolved_problem"] = resolve_problem(problem, config["common_params"])

    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuRiPP benchmark grids.")
    parser.add_argument(
        "--config",
        default="benchmark_config.json",
        help="Path to benchmark JSON config. Defaults to benchmark_config.json.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from saved results without running benchmarks.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run exactly one planned run by numeric run_index.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable output session name under output_root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit benchmark session directory, primarily for plot-only mode.",
    )
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> Path:
    if args.plot_only and args.run_id is not None:
        raise ValueError("--plot-only and --run-id are mutually exclusive")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    return config_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config_path = validate_cli_args(args)
        config = load_config(config_path)
        session_dir = resolve_session_dir(
            config,
            run_name=args.run_name,
            output_dir=args.output_dir,
            plot_only=args.plot_only,
        )
        if not args.plot_only:
            planned_runs = plan_runs(config)
            selected_runs = select_planned_runs(planned_runs, args.run_id)
            initialize_session(session_dir, config_path, config, planned_runs)
            if len(config["parallel"]["gpu_ids"]) > 1 and len(selected_runs) > 1:
                summary = run_parallel(selected_runs, session_dir, config)
            else:
                summary = run_sequential(selected_runs, session_dir, config)
            generate_per_run_plots(session_dir, config.get("plotting", {}))
            generate_aggregate_plots(session_dir, config.get("plotting", {}))
        else:
            summary = None
            planned_runs = []
            selected_runs = []
            generate_per_run_plots(session_dir)
            generate_aggregate_plots(session_dir)
    except Exception as exc:
        raise SystemExit(f"benchmark_runner: {exc}") from exc

    print(
        "benchmark_runner: execution complete; "
        f"planned {len(planned_runs)} run(s), selected {len(selected_runs)}; "
        f"session {session_dir}; summary {summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
