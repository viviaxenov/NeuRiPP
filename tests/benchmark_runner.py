"""Benchmark runner CLI skeleton.

This module intentionally avoids importing JAX at module import time. Worker
execution will set device-related environment variables before JAX is loaded in
later implementation stages.
"""

from __future__ import annotations

import argparse
import copy
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
ANALYTIC_KL_DISTRIBUTIONS = {"gaussian", "st"}

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
        raise ValueError(f"{name}.rhs.dim is not supported; dim is inferred from problem")
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


def _sample_data_distribution(
    distribution: dict[str, Any], common_params: dict[str, Any]
) -> Any:
    import numpy as np

    name = _distribution_name(distribution)
    n_samples = distribution.get("n_samples", common_params.get("N_samples", 16))
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("data distribution n_samples must be a positive integer")
    seed = distribution.get("seed", 3)
    rng = np.random.default_rng(seed)

    if name == "checkerboard":
        points = rng.uniform(size=(n_samples, 2))
        shifts_x = rng.integers(0, 4, size=n_samples) - 2
        shifts_y = rng.integers(0, 2, size=n_samples) * 2 + shifts_x % 2 - 2
        points[:, 0] += shifts_x
        points[:, 1] += shifts_y
        return points

    if name == "two_spirals":
        half = max(n_samples // 2, 1)
        n = np.sqrt(rng.uniform(size=(half, 1))) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.uniform(size=(half, 1)) * 0.5
        d1y = np.sin(n) * n + rng.uniform(size=(half, 1)) * 0.5
        points = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        points += rng.uniform(size=points.shape) * 0.1
        return points[:n_samples]

    if name == "eight_gaussians":
        theta = np.linspace(0.0, 2.0 * np.pi, 8)
        centers = 4.0 * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
        blob = rng.normal(size=(n_samples, 2)) * 0.5
        shift_ids = rng.integers(0, 7, size=n_samples)
        return (blob + centers[shift_ids, :]) / 1.414

    supported = ", ".join(sorted(DATA_DISTRIBUTIONS))
    raise ValueError(
        f"Unsupported data distribution {name!r}. Supported data distributions: {supported}"
    )


def resolve_problem(problem: dict[str, Any], common_params: dict[str, Any]) -> dict[str, Any]:
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

    if name in DATA_DISTRIBUTIONS:
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
        supported = ", ".join(sorted(DATA_DISTRIBUTIONS))
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


def _expand_template(template: dict[str, Any], skip_keys: set[str] | None = None) -> list[dict[str, Any]]:
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
        for top_values, rhs_values in itertools.product(expanded_top_level, expanded_rhs):
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
        for kwargs in _expand_template(method, skip_keys={"method"}):
            expanded.append({"method": method_name, "method_kwargs": kwargs})
    return expanded


def _expand_problem(problem: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand problem.distribution.seed if it is a list, producing one problem dict per seed."""
    distribution = problem.get("distribution")
    if not isinstance(distribution, dict):
        return [copy.deepcopy(problem)]

    seed_value = distribution.get("seed")
    if not isinstance(seed_value, list):
        return [copy.deepcopy(problem)]

    if not seed_value:
        raise ValueError("problem.distribution.seed must not be an empty list")

    expanded: list[dict[str, Any]] = []
    for seed in seed_value:
        problem_copy = copy.deepcopy(problem)
        problem_copy["distribution"]["seed"] = seed
        expanded.append(problem_copy)
    return expanded


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
                "problem": copy.deepcopy(problem),
                "resolved_problem": copy.deepcopy(config["resolved_problem"]),
                "architecture": copy.deepcopy(architecture),
                "method": method_config["method"],
                "method_kwargs": method_kwargs,
            }
        )

    return planned_runs


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
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _append_batched_metrics(metric_values: Any, metrics: dict[str, list[Any]]) -> None:
    import numpy as np

    names = ["loss", "grad_norm", "natural_grad_norm"]
    for name, value in zip(names, metric_values, strict=False):
        metrics[name].append(np.asarray(value, dtype=float))


def _metric_arrays_for_lane(arrays: dict[str, Any], index: int) -> dict[str, Any]:
    lane_arrays: dict[str, Any] = {}
    for name, values in arrays.items():
        lane_arrays[name] = values[:, index] if values.ndim > 1 else values
    return lane_arrays


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


def assigned_gpu_id(parallel_config: dict[str, Any], worker_id: int) -> int | str | None:
    gpu_ids = parallel_config.get("gpu_ids")
    if not gpu_ids:
        return None
    return gpu_ids[worker_id % len(gpu_ids)]


def setup_worker_environment(
    parallel_config: dict[str, Any], worker_id: int
) -> int | str | None:
    """Set worker environment before JAX is imported."""

    requested_gpu_id = assigned_gpu_id(parallel_config, worker_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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

    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    import jax
    import jax.numpy as jnp
    from flax import nnx

    from neuripp.functionals.CrossEntropy import cross_entropy
    from neuripp.functionals.KL import getKL, logpdf_st
    from neuripp.functionals.MMD import (
        checkerboard_generator,
        eight_gaussians_generator,
        getMMD,
        two_spirals_generator,
    )
    from neuripp.methods.anderson import get_anderson
    from neuripp.methods.ngd import get_ngd
    from neuripp.methods.optax_optimizer import get_optax, optax_optimizers
    from neuripp.methods.sgd import get_sgd
    from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward

    test_rhs = import_module("test_rhs")

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
        "logpdf_st": logpdf_st,
        "getMMD": getMMD,
        "data_generators": {
            "checkerboard": checkerboard_generator,
            "two_spirals": two_spirals_generator,
            "eight_gaussians": eight_gaussians_generator,
        },
        "rhs_registry": {
            "MLP": test_rhs.MLP,
            "LinearRHS": test_rhs.LinearRHS,
        },
        "activation_registry": activation_registry,
        "method_registry": {
            "sgd": get_sgd,
            "ngd": get_ngd,
            "anderson": get_anderson,
            **{name: get_optax for name in optax_optimizers if name != "sgd"},
        },
        "optax_methods": set(optax_optimizers) - {"sgd"},
    }


def build_rhs(
    architecture: dict[str, Any], dim: int, deps: dict[str, Any] | None = None
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

    return rhs_registry[rhs_model_name](dim, **rhs_kwargs)


def split_architecture_kwargs(
    architecture: dict[str, Any]
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
    deps: dict[str, Any] | None = None,
) -> Any:
    deps = deps or import_runtime_dependencies()
    _, ode_kwargs = split_architecture_kwargs(architecture)
    rhs = build_rhs(architecture, dim, deps=deps)
    return deps["ParametricPushforward"](
        rhs,
        architecture["N_monte_carlo"],
        architecture["seed"],
        ode_nstep_max=architecture["ode_nstep_max"],
        ode_method=architecture["ode_method"],
        divergence_method=architecture["divergence_method"],
        ode_kwargs=ode_kwargs,
    )


def build_data_generator(
    distribution: dict[str, Any], batch_size: int, deps: dict[str, Any] | None = None
) -> Any:
    deps = deps or import_runtime_dependencies()
    name = _distribution_name(distribution)
    data_generators = deps["data_generators"]
    if name not in data_generators:
        supported = ", ".join(sorted(data_generators))
        raise ValueError(
            f"Unsupported data distribution {name!r}; expected one of: {supported}"
        )
    return data_generators[name](
        batch_size,
        distribution.get("resample_each", 1),
        distribution.get("seed", 3),
    )


def build_gaussian_logpdf(dim: int, distribution: dict[str, Any], deps: dict[str, Any]) -> Any:
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


def build_problem(
    problem: dict[str, Any],
    resolved_problem: dict[str, Any],
    method_kwargs: dict[str, Any],
    deps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deps = deps or import_runtime_dependencies()
    kind = problem["functional"]["kind"]
    distribution = problem["distribution"]
    batch_size = method_kwargs["N_samples"]

    if kind == "CrossEntropy":
        return {
            "loss": deps["cross_entropy"],
            "data_gen": build_data_generator(distribution, batch_size, deps=deps),
        }

    if kind == "MMD":
        data_gen = build_data_generator(distribution, batch_size, deps=deps)
        bandwidth_batch = next(data_gen)
        bw_multipliers = problem["functional"].get("bw_multipliers")
        if bw_multipliers is not None:
            bw_multipliers = deps["jnp"].asarray(bw_multipliers)
        return {
            "loss": deps["getMMD"](
                batch_size, data=bandwidth_batch, bw_multipliers=bw_multipliers
            ),
            "data_gen": data_gen,
        }

    if kind == "KL":
        name = _distribution_name(distribution)
        if name == "st":
            logpdf = deps["logpdf_st"]
        elif name == "gaussian":
            logpdf = build_gaussian_logpdf(resolved_problem["dim"], distribution, deps)
        else:
            raise ValueError(f"Unsupported KL distribution {name!r}")
        return {"loss": deps["getKL"](logpdf, batch_size), "data_gen": None}

    raise ValueError(f"Unsupported functional kind {kind!r}")


METHOD_EXECUTION_KEYS = {
    "max_iterations",
    "N_samples",
    "method_seed",
    "plot_every",
}


def _method_specific_kwargs(method_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in method_kwargs.items()
        if key not in METHOD_EXECUTION_KEYS
    }


def build_method(
    method_name: str,
    method_kwargs: dict[str, Any],
    loss: Any,
    deps: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    deps = deps or import_runtime_dependencies()
    method_registry = deps["method_registry"]
    if method_name not in method_registry:
        supported = ", ".join(sorted(method_registry))
        raise ValueError(f"Unsupported method {method_name!r}; expected one of: {supported}")

    kwargs = _method_specific_kwargs(method_kwargs)
    if method_name in deps["optax_methods"]:
        return method_registry[method_name](loss, method_name, **kwargs)
    return method_registry[method_name](loss, **kwargs)


def build_group_problem(
    planned_runs: list[dict[str, Any]], deps: dict[str, Any] | None = None
) -> dict[str, Any]:
    deps = deps or import_runtime_dependencies()
    representative = planned_runs[0]
    problem = representative["problem"]
    resolved_problem = representative["resolved_problem"]
    method_kwargs = representative["method_kwargs"]
    functional_kind = problem["functional"]["kind"]
    per_seed_batch_size = method_kwargs["N_samples"]
    n_seeds = len(planned_runs)

    if functional_kind == "KL":
        return build_problem(problem, resolved_problem, method_kwargs, deps=deps)

    effective_batch_size = n_seeds * per_seed_batch_size
    distribution = problem["distribution"]
    data_gen = build_data_generator(distribution, effective_batch_size, deps=deps)

    if functional_kind == "CrossEntropy":
        return {"loss": deps["cross_entropy"], "data_gen": data_gen}

    if functional_kind == "MMD":
        bandwidth_batch = next(data_gen).reshape(
            n_seeds, per_seed_batch_size, resolved_problem["dim"]
        )[0]
        bw_multipliers = problem["functional"].get("bw_multipliers")
        if bw_multipliers is not None:
            bw_multipliers = deps["jnp"].asarray(bw_multipliers)
        return {
            "loss": deps["getMMD"](
                per_seed_batch_size,
                data=bandwidth_batch,
                bw_multipliers=bw_multipliers,
            ),
            "data_gen": data_gen,
        }

    raise ValueError(f"Unsupported functional kind {functional_kind!r}")


def _split_carry_for_vmap(carry: Any, deps: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    nnx = deps["nnx"]
    specs: list[Any] = []
    dynamic: list[Any] = []
    for item in carry:
        if isinstance(item, nnx.Module):
            graphdef, state = nnx.split(item)
            specs.append(("module", graphdef))
            dynamic.append(state)
        else:
            specs.append(("value", None))
            dynamic.append(item)
    return tuple(specs), tuple(dynamic)


def _merge_carry_from_vmap(specs: tuple[Any, ...], dynamic: tuple[Any, ...], deps: dict[str, Any]) -> Any:
    nnx = deps["nnx"]
    carry_items: list[Any] = []
    for spec, item in zip(specs, dynamic, strict=True):
        kind, graphdef = spec
        if kind == "module":
            carry_items.append(nnx.merge(graphdef, item))
        else:
            carry_items.append(item)
    return tuple(carry_items)


def _stack_trees(items: list[Any], deps: dict[str, Any]) -> Any:
    return deps["jax"].tree.map(lambda *xs: deps["jnp"].stack(xs), *items)


def _lane_from_tree(tree: Any, index: int, deps: dict[str, Any]) -> Any:
    return deps["jax"].tree.map(lambda x: x[index], tree)


def _model_from_batched_carry(
    carry_specs: tuple[Any, ...],
    batched_carry_dynamic: tuple[Any, ...],
    index: int,
    deps: dict[str, Any],
) -> Any:
    carry_dynamic = _lane_from_tree(batched_carry_dynamic, index, deps)
    carry = _merge_carry_from_vmap(carry_specs, carry_dynamic, deps)
    return _model_from_carry(carry)


def execute_seed_group(
    planned_runs: list[dict[str, Any]],
    session_dir: Path,
    worker_id: int,
    device_info: dict[str, Any],
    deps: dict[str, Any] | None = None,
    progress_position: int | None = None,
) -> dict[str, Any]:
    deps = deps or import_runtime_dependencies()
    representative = planned_runs[0]
    run_dirs = [session_dir / "runs" / planned_run["run_id"] for planned_run in planned_runs]
    for run_dir, planned_run in zip(run_dirs, planned_runs, strict=True):
        run_dir.mkdir(parents=True, exist_ok=True)
        write_expanded_run_config(run_dir, planned_run)
        write_status(
            run_dir,
            "running",
            planned_run,
            worker_id,
            device_info,
            started_at=_utc_now_iso(),
        )

    try:
        jax = deps["jax"]
        jnp = deps["jnp"]
        problem_state = build_group_problem(planned_runs, deps=deps)
        models = [
            build_model(
                planned_run["architecture"],
                planned_run["method_kwargs"],
                planned_run["resolved_problem"]["dim"],
                deps=deps,
            )
            for planned_run in planned_runs
        ]
        init_fun, step_fun = build_method(
            representative["method"],
            representative["method_kwargs"],
            problem_state["loss"],
            deps=deps,
        )

        nnx = deps["nnx"]
        graphdef, first_state = nnx.split(models[0])
        model_states = [first_state]
        model_states.extend(nnx.split(model)[1] for model in models[1:])
        batched_model_state = _stack_trees(model_states, deps)

        data_gen = problem_state["data_gen"]
        n_seeds = len(planned_runs)
        per_seed_batch_size = representative["method_kwargs"]["N_samples"]
        dim = representative["resolved_problem"]["dim"]

        def reshape_data_batch(flat_batch: Any) -> Any:
            return jnp.asarray(flat_batch).reshape(n_seeds, per_seed_batch_size, dim)

        def init_one(model_state: Any, batch: Any | None = None) -> tuple[Any, ...]:
            model = nnx.merge(graphdef, model_state)
            if batch is None:
                carry = init_fun(model)
            else:
                carry = init_fun(model, batch)
            _, carry_dynamic = _split_carry_for_vmap(carry, deps)
            return carry_dynamic

        if data_gen is not None:
            init_batch = reshape_data_batch(next(data_gen))
            batched_carry_dynamic = jax.vmap(init_one, in_axes=(0, 0))(batched_model_state, init_batch)
        else:
            batched_carry_dynamic = jax.vmap(lambda model_state: init_one(model_state))(batched_model_state)

        carry_specs = None
        if data_gen is not None:
            carry_specs, _ = _split_carry_for_vmap(
                init_fun(models[0], init_batch[0]),
                deps,
            )
        else:
            carry_specs, _ = _split_carry_for_vmap(init_fun(models[0]), deps)

        def step_one(carry_dynamic: tuple[Any, ...], batch: Any | None = None) -> tuple[Any, Any]:
            carry = _merge_carry_from_vmap(carry_specs, carry_dynamic, deps)
            if batch is None:
                next_carry, values = step_fun(carry)
            else:
                next_carry, values = step_fun(carry, batch)
            _, next_dynamic = _split_carry_for_vmap(next_carry, deps)
            return next_dynamic, values

        if data_gen is not None:
            batched_step_fun = nnx.jit(jax.vmap(step_one, in_axes=(0, 0)))
        else:
            batched_step_fun = nnx.jit(jax.vmap(lambda carry_dynamic: step_one(carry_dynamic)))

        metrics: dict[str, list[Any]] = {
            "loss": [],
            "grad_norm": [],
            "natural_grad_norm": [],
        }
        best_losses = [float("inf")] * n_seeds
        max_iterations = representative["method_kwargs"]["max_iterations"]
        plot_every = representative["method_kwargs"].get("plot_every")
        session_plot_dir = session_dir / "plots"
        session_plot_dir.mkdir(parents=True, exist_ok=True)
        run_plot_paths = [session_plot_dir / f"{planned_run['run_id']}_plots.pdf" for planned_run in planned_runs]

        progress_bar = None
        if progress_position is not None:
            from tqdm import tqdm

            desc = (
                f"worker {worker_id} {device_info['device_label']} "
                f"{representative['run_id']} x{n_seeds} {representative['method']}"
            )
            progress_bar = tqdm(
                total=max_iterations,
                desc=desc,
                position=progress_position,
                leave=False,
            )

        for iteration in range(max_iterations):
            if data_gen is not None:
                step_batch = reshape_data_batch(next(data_gen))
                batched_carry_dynamic, values = batched_step_fun(batched_carry_dynamic, step_batch)
            else:
                batched_carry_dynamic, values = batched_step_fun(batched_carry_dynamic)
            _append_batched_metrics(values, metrics)

            import numpy as np

            current_losses = np.asarray(values[0], dtype=float)
            for index, current_loss in enumerate(current_losses):
                current_loss_value = float(current_loss)
                if current_loss_value < best_losses[index]:
                    best_losses[index] = current_loss_value
                    write_checkpoint(
                        run_dirs[index] / "checkpoints" / "best",
                        _model_from_batched_carry(
                            carry_specs,
                            batched_carry_dynamic,
                            index,
                            deps,
                        ),
                        deps,
                        planned_runs[index],
                        "best",
                        metric_value=current_loss_value,
                    )

            if progress_bar is not None:
                progress_bar.update(1)

            if plot_every is not None and (iteration + 1) % plot_every == 0:
                arrays = _metric_arrays(metrics)
                for index, planned_run in enumerate(planned_runs):
                    plot_run_diagnostics(
                        _model_from_batched_carry(
                            carry_specs,
                            batched_carry_dynamic,
                            index,
                            deps,
                        ),
                        _metric_arrays_for_lane(arrays, index),
                        f"{planned_run['run_id']}, iteration {iteration + 1}",
                        run_plot_paths[index],
                        n_samples=per_seed_batch_size,
                    )

        if progress_bar is not None:
            progress_bar.close()

        arrays = _metric_arrays(metrics)
        results = []
        import numpy as np

        for index, planned_run in enumerate(planned_runs):
            lane_arrays = _metric_arrays_for_lane(arrays, index)
            np.savez(run_dirs[index] / "arrays.npz", **lane_arrays)
            model = _model_from_batched_carry(carry_specs, batched_carry_dynamic, index, deps)
            plot_run_diagnostics(
                model,
                lane_arrays,
                f"{planned_run['run_id']}, final",
                run_plot_paths[index],
                n_samples=per_seed_batch_size,
            )
            last_loss = float(lane_arrays["loss"][-1]) if "loss" in lane_arrays else None
            write_checkpoint(
                run_dirs[index] / "checkpoints" / "last",
                model,
                deps,
                planned_run,
                "last",
                metric_value=last_loss,
            )
            write_status(
                run_dirs[index],
                "success",
                planned_run,
                worker_id,
                device_info,
                finished_at=_utc_now_iso(),
                best_loss=best_losses[index],
                best_checkpoint_criterion="minimum_loss",
                arrays_path="arrays.npz",
            )
            results.append({"status": "success", "run_id": planned_run["run_id"], "run_dir": run_dirs[index]})
        return {
            "status": "success",
            "success": len(results),
            "failed": 0,
            "results": results,
        }
    except Exception as exc:
        if "progress_bar" in locals() and progress_bar is not None:
            progress_bar.close()
        traceback_text = traceback.format_exc()
        results = []
        for run_dir, planned_run in zip(run_dirs, planned_runs, strict=True):
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
            )
            results.append(
                {
                    "status": "failed",
                    "run_id": planned_run["run_id"],
                    "run_dir": run_dir,
                    "error": str(exc),
                }
            )
        return {
            "status": "failed",
            "success": 0,
            "failed": len(results),
            "results": results,
        }


def execute_task(
    task: list[dict[str, Any]],
    session_dir: Path,
    worker_id: int,
    device_info: dict[str, Any],
    deps: dict[str, Any] | None = None,
    progress_position: int | None = None,
) -> dict[str, Any]:
    if len(task) == 1:
        result = execute_run(
            task[0],
            session_dir,
            worker_id,
            device_info,
            deps=deps,
            progress_position=progress_position,
        )
        return {
            "status": result["status"],
            "success": 1 if result["status"] == "success" else 0,
            "failed": 1 if result["status"] == "failed" else 0,
            "results": [result],
        }
    return execute_seed_group(
        task,
        session_dir,
        worker_id,
        device_info,
        deps=deps,
        progress_position=progress_position,
    )


def execute_run(
    planned_run: dict[str, Any],
    session_dir: Path,
    worker_id: int,
    device_info: dict[str, Any],
    deps: dict[str, Any] | None = None,
    progress_position: int | None = None,
) -> dict[str, Any]:
    deps = deps or import_runtime_dependencies()
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
    )

    try:
        problem_state = build_problem(
            planned_run["problem"],
            planned_run["resolved_problem"],
            planned_run["method_kwargs"],
            deps=deps,
        )
        model = build_model(
            planned_run["architecture"],
            planned_run["method_kwargs"],
            planned_run["resolved_problem"]["dim"],
            deps=deps,
        )
        init_fun, step_fun = build_method(
            planned_run["method"], planned_run["method_kwargs"], problem_state["loss"], deps=deps
        )
        step_fun = deps["nnx"].jit(step_fun)

        data_gen = problem_state["data_gen"]
        init_args = (next(data_gen),) if data_gen is not None else ()
        carry = init_fun(model, *init_args)
        metrics: dict[str, list[float]] = {
            "loss": [],
            "grad_norm": [],
            "natural_grad_norm": [],
        }
        best_loss = float("inf")
        max_iterations = planned_run["method_kwargs"].get("max_iterations")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("method_kwargs.max_iterations must be a positive integer")
        plot_every = planned_run["method_kwargs"].get("plot_every")
        if plot_every is not None and (
            not isinstance(plot_every, int) or plot_every < 1
        ):
            raise ValueError("method_kwargs.plot_every must be a positive integer when provided")
        session_plot_dir = session_dir / "plots"
        session_plot_dir.mkdir(parents=True, exist_ok=True)
        run_plot_path = session_plot_dir / f"{planned_run['run_id']}_plots.pdf"

        iteration_range = range(max_iterations)
        progress_bar = None
        if progress_position is not None:
            from tqdm import tqdm

            desc = (
                f"worker {worker_id} {device_info['device_label']} "
                f"{planned_run['run_id']} {planned_run['method']}"
            )
            progress_bar = tqdm(
                total=max_iterations,
                desc=desc,
                position=progress_position,
                leave=False,
            )

        for iteration in iteration_range:
            step_args = (next(data_gen),) if data_gen is not None else ()
            carry, values = step_fun(carry, *step_args)
            _append_metrics(values, metrics)
            current_loss = metrics["loss"][-1]
            if current_loss < best_loss:
                best_loss = current_loss
                write_checkpoint(
                    run_dir / "checkpoints" / "best",
                    _model_from_carry(carry),
                    deps,
                    planned_run,
                    "best",
                    metric_value=best_loss,
                )
            if progress_bar is not None:
                progress_bar.update(1)
            if plot_every is not None and (iteration + 1) % plot_every == 0:
                title = f"{planned_run['run_id']}, iteration {iteration + 1}"
                plot_run_diagnostics(
                    _model_from_carry(carry),
                    _metric_arrays(metrics),
                    title,
                    run_plot_path,
                    n_samples=planned_run["method_kwargs"]["N_samples"],
                )

        if progress_bar is not None:
            progress_bar.close()

        arrays = _metric_arrays(metrics)
        arrays_path = run_dir / "arrays.npz"
        import numpy as np

        np.savez(arrays_path, **arrays)
        plot_run_diagnostics(
            _model_from_carry(carry),
            arrays,
            f"{planned_run['run_id']}, final",
            run_plot_path,
            n_samples=planned_run["method_kwargs"]["N_samples"],
        )
        write_checkpoint(
            run_dir / "checkpoints" / "last",
            _model_from_carry(carry),
            deps,
            planned_run,
            "last",
            metric_value=metrics["loss"][-1] if metrics["loss"] else None,
        )
        write_status(
            run_dir,
            "success",
            planned_run,
            worker_id,
            device_info,
            finished_at=_utc_now_iso(),
            best_loss=best_loss,
            best_checkpoint_criterion="minimum_loss",
            arrays_path="arrays.npz",
        )
        return {"status": "success", "run_id": planned_run["run_id"], "run_dir": run_dir}
    except Exception as exc:
        if "progress_bar" in locals() and progress_bar is not None:
            progress_bar.close()
        traceback_text = traceback.format_exc()
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
        )
        return {
            "status": "failed",
            "run_id": planned_run["run_id"],
            "run_dir": run_dir,
            "error": str(exc),
        }


def run_sequential(
    selected_runs: list[dict[str, Any]],
    session_dir: Path,
    config: dict[str, Any],
) -> dict[str, int]:
    worker_id = 0
    requested_gpu_id = setup_worker_environment(config["parallel"], worker_id)
    device_info = get_worker_device_info(requested_gpu_id, config.get("config"))
    deps = import_runtime_dependencies()
    summary = {"success": 0, "failed": 0, "total": len(selected_runs)}
    tasks = group_runs_for_seed_vmap(selected_runs)
    from tqdm import tqdm

    with tqdm(total=len(selected_runs), desc="Runs", position=0, leave=True) as runs_pbar:
        for task in tasks:
            result = execute_task(
                task,
                session_dir,
                worker_id,
                device_info,
                deps=deps,
                progress_position=worker_id + 1,
            )
            summary["success"] += result["success"]
            summary["failed"] += result["failed"]
            runs_pbar.update(len(result["results"]))
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
            task = task_queue.get()
            if task is None:
                message_queue.put(
                    {"event": "worker_empty_queue", "worker_id": worker_id}
                )
                break

            run_ids = [planned_run["run_id"] for planned_run in task]
            message_queue.put(
                {
                    "event": "run_started",
                    "worker_id": worker_id,
                    "run_id": run_ids[0],
                    "run_count": len(run_ids),
                }
            )
            result = execute_task(
                task,
                session_path,
                worker_id,
                device_info,
                deps=deps,
                progress_position=worker_id + 1,
            )
            for run_result in result["results"]:
                if run_result["status"] == "success":
                    message_queue.put(
                        {
                            "event": "run_success",
                            "worker_id": worker_id,
                            "run_id": run_result["run_id"],
                        }
                    )
                else:
                    message_queue.put(
                        {
                            "event": "run_error",
                            "worker_id": worker_id,
                            "run_id": run_result["run_id"],
                            "error": run_result.get("error", "unknown error"),
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

    tasks = group_runs_for_seed_vmap(selected_runs)
    n_workers = min(config["parallel"]["n_workers"], len(tasks))
    if n_workers <= 1:
        return run_sequential(selected_runs, session_dir, config)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    message_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
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
    terminal_events = {"run_success", "run_error"}

    with tqdm(total=len(selected_runs), desc="Runs", position=0, leave=True) as runs_pbar:
        while len(exited_workers) < n_workers:
            try:
                message = message_queue.get(timeout=0.2)
            except queue.Empty:
                for worker_id, process in enumerate(processes):
                    if worker_id in exited_workers:
                        continue
                    if not process.is_alive() and process.exitcode not in {None, 0}:
                        exited_workers.add(worker_id)
                        summary["failed"] += 1
                        tqdm.write(
                            f"worker {worker_id} exited unexpectedly with code {process.exitcode}"
                        )
                continue

            event = message.get("event")
            worker_id = message.get("worker_id")
            if event == "worker_started":
                tqdm.write(
                    f"worker {worker_id} started on {message.get('device_label')} "
                    f"devices={message.get('jax_devices')}"
                )
            elif event == "run_started":
                run_count = message.get("run_count", 1)
                if run_count > 1:
                    tqdm.write(
                        f"worker {worker_id} started {message.get('run_id')} x{run_count}"
                    )
                else:
                    tqdm.write(
                        f"worker {worker_id} started {message.get('run_id')}"
                    )
            elif event in terminal_events:
                if event == "run_success":
                    summary["success"] += 1
                else:
                    summary["failed"] += 1
                    tqdm.write(
                        f"worker {worker_id} failed {message.get('run_id')}: "
                        f"{message.get('error')}"
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

    _write_json(session_dir / "summary.json", summary)
    return summary


def load_completed_runs(session_dir: str | Path) -> list[dict[str, Any]]:
    """Load successful benchmark runs from a session directory.

    Failed and incomplete runs are skipped. If no successful runs are present,
    an empty list is returned.
    """

    session_dir = Path(session_dir)
    runs_dir = session_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Benchmark runs directory does not exist: {runs_dir}")

    completed_runs: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        status_path = run_dir / "status.json"
        expanded_config_path = run_dir / "expanded_config.json"
        arrays_path = run_dir / "arrays.npz"
        if not status_path.exists() or not expanded_config_path.exists():
            continue

        status = _load_json(status_path)
        if status.get("status") != "success":
            continue
        if not arrays_path.exists():
            continue

        import numpy as np

        with np.load(arrays_path) as loaded_arrays:
            arrays = {name: loaded_arrays[name] for name in loaded_arrays.files}

        expanded_config = _load_json(expanded_config_path)
        completed_runs.append(
            {
                "run_id": expanded_config.get("run_id", run_dir.name),
                "run_dir": run_dir,
                "status": status,
                "expanded_config": expanded_config,
                "arrays": arrays,
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
    if "method_kwargs" in expanded_config:
        _flatten_mapping("method_kwargs", expanded_config["method_kwargs"], flat)
    if "architecture" in expanded_config:
        _flatten_mapping("architecture", expanded_config["architecture"], flat)
    if "problem" in expanded_config:
        _flatten_mapping("problem", expanded_config["problem"], flat)
    return flat


def varying_param_keys(expanded_configs: list[dict[str, Any]]) -> list[str]:
    flattened = [flatten_run_params(config) for config in expanded_configs]
    keys: list[str] = []
    for flat in flattened:
        for key in flat:
            if key not in keys:
                keys.append(key)

    varying: list[str] = []
    for key in keys:
        values = {
            _value_identity(flat[key])
            for flat in flattened
            if key in flat
        }
        if len(values) > 1:
            varying.append(key)
    return varying


def _display_param_key(key: str) -> str:
    for prefix in ("method_kwargs.", "architecture.", "problem."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def format_run_label(expanded_config: dict[str, Any], keys: list[str]) -> str:
    flat = flatten_run_params(expanded_config)
    parts = [f"{_display_param_key(key)} {flat[key]}" for key in keys if key in flat]
    return ", ".join(parts) if parts else expanded_config.get("run_id", "run")


def _as_expanded_config(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("expanded_config", item)


def _loss_array(item: dict[str, Any]) -> Any:
    arrays = item.get("arrays")
    if arrays is not None and "loss" in arrays:
        return arrays["loss"]
    if "loss" in item:
        return item["loss"]
    expanded_config = item.get("expanded_config")
    if isinstance(expanded_config, dict):
        arrays = expanded_config.get("arrays")
        if arrays is not None and "loss" in arrays:
            return arrays["loss"]
        if "loss" in expanded_config:
            return expanded_config["loss"]
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


def _group_items_by_keys(
    items: list[dict[str, Any]], keys: list[str]
) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    order: list[tuple[str, ...]] = []
    for item in items:
        flat = flatten_run_params(_as_expanded_config(item))
        group_key = tuple(_value_identity(flat.get(key)) for key in keys)
        if group_key not in groups:
            groups[group_key] = []
            order.append(group_key)
        groups[group_key].append(item)
    return [(key, groups[key]) for key in order]


def _is_seed_key(key: str) -> bool:
    leaf = key.rsplit(".", 1)[-1].lower()
    return leaf == "seed" or leaf.endswith("_seed")


def _is_groupable_seed_key(key: str) -> bool:
    return _is_seed_key(key) and key != "problem.distribution.seed"


def _seed_group_identity(planned_run: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    flat = flatten_run_params(planned_run)
    return tuple(
        (key, _value_identity(value))
        for key, value in flat.items()
        if not _is_groupable_seed_key(key)
    )


def group_runs_for_seed_vmap(
    selected_runs: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    order: list[tuple[tuple[str, str], ...]] = []
    for planned_run in selected_runs:
        group_key = _seed_group_identity(planned_run)
        if group_key not in groups:
            groups[group_key] = []
            order.append(group_key)
        groups[group_key].append(planned_run)
    return [groups[group_key] for group_key in order]


def _filename_suffix(index: int, group_key: dict[str, Any]) -> str:
    if not group_key:
        return ""
    return f"arch_{index:03d}"


def _representative_params(items: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    base = copy.deepcopy(_as_expanded_config(items[0]))
    flat = flatten_run_params(base)
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
    representatives: list[dict[str, Any]], style_channels: dict[str, Any]
) -> list[dict[str, Any]]:
    if not representatives:
        return []

    flattened = []
    for representative in representatives:
        line_params = representative.get("_line_params")
        if isinstance(line_params, dict):
            flattened.append(line_params)
        else:
            flattened.append(flatten_run_params(representative))

    varying_keys = _varying_keys_from_flattened(flattened)
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


def get_lines(expanded_configs: list[dict[str, Any]], style_channels: dict[str, Any]) -> list[dict[str, Any]]:
    """Build aggregate plotting groups and loss series definitions.

    Architecture-varying parameters define top-level plot groups. Seed-varying
    parameters (architecture.seed, method_kwargs.method_seed,
    problem.distribution.seed, etc.) are aggregated into mean +/- std loss
    tubes within each architecture group. Seed keys are excluded from
    architecture grouping and line identity so configs that differ only by
    seed are treated as the same architecture/problem/method.
    """

    if not expanded_configs:
        return []

    import numpy as np

    flattened_all = [flatten_run_params(_as_expanded_config(item)) for item in expanded_configs]
    varying_keys_all = _varying_keys_from_flattened(flattened_all)
    architecture_keys = [
        key
        for key in varying_keys_all
        if key.startswith("architecture.") and not _is_seed_key(key)
    ]
    architecture_groups = _group_items_by_keys(expanded_configs, architecture_keys)
    groups: list[dict[str, Any]] = []

    for group_index, (_, group_items) in enumerate(architecture_groups):
        group_flat = [flatten_run_params(_as_expanded_config(item)) for item in group_items]
        architecture_group_key = {
            key: group_flat[0][key]
            for key in architecture_keys
            if key in group_flat[0]
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
        seed_keys = [key for key in varying_keys if _is_seed_key(key)]
        line_param_keys = [
            key
            for key in varying_keys
            if key not in architecture_keys and key not in seed_keys
        ]

        line_items: list[dict[str, Any]] = []
        representatives: list[dict[str, Any]] = []
        if seed_keys:
            for _, seed_group_items in _group_items_by_keys(group_items, line_param_keys):
                losses = [np.asarray(_loss_array(item), dtype=float) for item in seed_group_items]
                min_length = min(loss.shape[0] for loss in losses)
                if any(loss.shape[0] != min_length for loss in losses):
                    warnings.warn(
                        "Loss arrays in a seed group have different lengths; truncating to minimum length.",
                        stacklevel=2,
                    )
                stacked = np.stack([loss[:min_length] for loss in losses], axis=0)
                mean = stacked.mean(axis=0)
                std = stacked.std(axis=0)
                representative = _representative_params(seed_group_items, line_param_keys)
                representatives.append(representative)
                line_items.append(
                    {
                        "label": _line_label(representative, line_param_keys),
                        "loss_mean": mean,
                        "loss_upper": mean + std,
                        "loss_lower": mean - std,
                    }
                )
        else:
            for item in group_items:
                representative = _representative_params([item], line_param_keys)
                representatives.append(representative)
                line_items.append(
                    {
                        "label": _line_label(representative, line_param_keys)
                        or _as_expanded_config(item).get("run_id", "run"),
                        "loss": np.asarray(_loss_array(item), dtype=float),
                    }
                )

        styles = _assign_line_styles(representatives, style_channels or {})
        for line, style in zip(line_items, styles, strict=True):
            line["style"] = style

        groups.append(
            {
                "group_key": architecture_group_key,
                "title": title,
                "filename_suffix": _filename_suffix(group_index, architecture_group_key),
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
    plotting = plotting or {}
    completed_runs = load_completed_runs(session_dir)
    if not completed_runs:
        return []

    style_channels = plotting.get("style_channels", {})
    groups = get_lines(completed_runs, style_channels)
    if not groups:
        return []

    plot_dir = session_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for group in groups:
        suffix = group['filename_suffix']
        filename = f"aggregate_loss_{suffix}.pdf" if suffix else "aggregate_loss.pdf"
        output_path = plot_dir / filename
        _plot_aggregate_group(group, output_path)
        written_paths.append(output_path)

    return written_paths


def plot_run_diagnostics(
    model: Any,
    arrays: dict[str, Any],
    title: str,
    output_path: Path,
    n_samples: int = 512,
) -> None:
    import matplotlib.pyplot as plt

    x = model.sample(n_samples)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 8), layout="constrained")

    ax = axs[0]
    if x.shape[-1] >= 2:
        ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}(z)$", marker="*", s=5.0)
    else:
        ax.hist(x[:, 0], bins=40, label=r"$T_{\text{opt}}(z)$")
    ax.legend()

    ax = axs[1]
    if "loss" in arrays:
        ax.plot(arrays["loss"], label="Loss")
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
        ax.grid()
    fig.suptitle(title)
    fig.savefig(output_path)
    plt.close(fig)


def generate_per_run_plots(
    session_dir: str | Path,
    plotting: dict[str, Any] | None = None,
) -> list[Path]:
    session_dir = Path(session_dir)
    plotting = plotting or {}
    completed_runs = load_completed_runs(session_dir)
    if not completed_runs:
        return []

    plot_dir = session_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    planned_runs_path = session_dir / "planned_runs.json"
    if planned_runs_path.exists():
        expanded_configs = _load_json_value(planned_runs_path)
    else:
        expanded_configs = [run["expanded_config"] for run in completed_runs]
    varying_keys = varying_param_keys(expanded_configs)
    n_samples = plotting.get("n_samples_plot", plotting.get("n_samples", 512))
    written_paths: list[Path] = []

    for run in completed_runs:
        expanded_config = run["expanded_config"]
        model = load_model_checkpoint(session_dir, run["run_id"], key="last")
        title = format_run_label(expanded_config, varying_keys)
        output_path = plot_dir / f"{run['run_id']}_plots.pdf"
        plot_run_diagnostics(model, run["arrays"], title, output_path, n_samples=n_samples)
        written_paths.append(output_path)

    return written_paths


def load_model_checkpoint(
    session_dir: str | Path,
    run_id: str,
    key: str = "last",
) -> Any:
    if key not in {"last", "best"}:
        raise ValueError("checkpoint key must be either 'last' or 'best'")

    session_dir = Path(session_dir)
    run_dir = session_dir / "runs" / run_id
    expanded_config_path = run_dir / "expanded_config.json"
    if not expanded_config_path.exists():
        raise FileNotFoundError(f"Unknown run_id or missing expanded config: {run_id}")

    checkpoint_dir = (run_dir / "checkpoints" / key).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    planned_run = _load_json(expanded_config_path)
    deps = import_runtime_dependencies()
    model = build_model(
        planned_run["architecture"],
        planned_run["method_kwargs"],
        planned_run["resolved_problem"]["dim"],
        deps=deps,
    )

    import orbax.checkpoint as ocp

    graphdef, state = deps["nnx"].split(model)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(checkpoint_dir, target=state)
    return deps["nnx"].merge(graphdef, state)


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
    _require_object(config["common_params"], "common_params")
    problem = _require_object(config["problem"], "problem")
    architectures = _require_non_empty_object_list(config["architectures"], "architectures")
    methods = _require_non_empty_object_list(config["methods"], "methods")
    _require_object(config["plotting"], "plotting")

    functional = _require_object(problem.get("functional"), "problem.functional")
    if "name" in functional:
        raise ValueError("problem.functional.name is not supported; use functional.kind")
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

    parallel = config.setdefault("parallel", {})
    _require_object(parallel, "parallel")
    n_workers = parallel.setdefault("n_workers", 1)
    if not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError("parallel.n_workers must be an integer greater than or equal to 1")
    if "gpu_ids" in parallel:
        gpu_ids = parallel["gpu_ids"]
        if not isinstance(gpu_ids, list) or not gpu_ids:
            raise ValueError("parallel.gpu_ids must be a non-empty list when provided")
        for index, gpu_id in enumerate(gpu_ids):
            if not isinstance(gpu_id, (int, str)):
                raise ValueError(
                    f"parallel.gpu_ids[{index}] must be an integer or string device id"
                )

    jax_config = config.setdefault("config", [])
    if not isinstance(jax_config, list):
        raise ValueError("config must be a list")
    for index, entry in enumerate(jax_config):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"config[{index}] must be a [flag, value] pair"
            )
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
            if config["parallel"]["n_workers"] > 1 and len(selected_runs) > 1:
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
