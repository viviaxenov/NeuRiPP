"""JAX-free JSON configuration validation and deterministic run planning."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from image_benchmarks.datasets.registry import get_dataset_spec


REQUIRED_TOP_LEVEL = {
    "experiment",
    "problem",
    "rhs",
    "training",
    "methods",
    "evaluation",
    "resources",
}
OPTIONAL_TOP_LEVEL = {"plotting"}
METHOD_NAMES = {
    "adagrad",
    "adam",
    "adamw",
    "anderson",
    "lion",
    "ngd",
    "rmsprop",
    "sgd",
    "yogi",
}
COMPUTE_DTYPES = {"bfloat16", "float16", "float32"}
MLP_ACTIVATIONS = {"gelu", "relu", "selu", "silu", "swish", "tanh"}
UNET_VARIANTS = {"small", "cifar_reference", "large"}
SIT_VARIANTS = {"S", "B", "L", "XL"}
SAMPLING_METHODS = {"rk45", "euler", "heun"}
RNG_STREAMS = (
    "dataset_shuffle",
    "augmentation",
    "encoder_sampling",
    "fm_noise",
    "fm_time",
    "model_dropout",
    "sampling",
    "evaluation",
)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer_at_least(value: Any, minimum: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_number_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not float(item) > 0.0
        for item in value
    ):
        raise ValueError(f"{name} values must be positive numbers")
    return [float(item) for item in value]


def _resolve_path(value: Any, base: Path, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _seed(master_seed: int, restart_index: int, stream: str) -> int:
    digest = hashlib.sha256(
        f"{master_seed}:{restart_index}:{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in stack:
        raise ValueError(f"Config inheritance cycle: {' -> '.join(map(str, (*stack, path)))}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Image benchmark config must contain a JSON object")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    if not isinstance(parent, str) or not parent:
        raise ValueError("extends must be a non-empty relative JSON path")
    parent_path = (path.parent / parent).resolve()
    return _deep_merge(_read_with_extends(parent_path, (*stack, path)), payload)


def _validate_encoder(
    config: dict[str, Any], image_shape: tuple[int, int, int], base: Path
) -> tuple[int, ...]:
    encoder_type = config.get("type")
    if encoder_type == "none":
        return image_shape
    if encoder_type == "ae":
        latent_dim = _positive_integer(config.get("latent_dim"), "problem.encoder.latent_dim")
        checkpoint = _resolve_path(
            config.get("checkpoint"), base, "problem.encoder.checkpoint"
        )
        config["checkpoint"] = checkpoint
        if not Path(checkpoint).exists() and not config.get("train_if_missing", False):
            raise FileNotFoundError(
                f"AE checkpoint is missing and train_if_missing is false: {checkpoint}"
            )
        if config.get("frozen_during_flow_training", True) is not True:
            raise ValueError("AE must be frozen during Flow Matching training")
        if config.get("cache_latents", True):
            config["latent_cache_dir"] = _resolve_path(
                config.get("latent_cache_dir", "artifacts/latents"),
                base,
                "problem.encoder.latent_cache_dir",
            )
        return (latent_dim,)
    if encoder_type != "vae":
        raise ValueError("problem.encoder.type must be one of: none, ae, vae")
    obsolete = {"source_dir", "source_auto_download"} & set(config)
    if obsolete:
        raise ValueError(
            "Packaged DiffuseNNX does not accept obsolete VAE source fields: "
            + ", ".join(sorted(obsolete))
        )
    _boolean(config.get("sample_posterior", True), "problem.encoder.sample_posterior")
    if image_shape[-1] != 3 or image_shape[0] % 8 or image_shape[1] % 8:
        raise ValueError("VAE requires RGB image dimensions divisible by eight")
    checkpoint = _resolve_path(
        config.get("checkpoint"), base, "problem.encoder.checkpoint"
    )
    config["checkpoint"] = checkpoint
    if not Path(checkpoint).exists() and not config.get("auto_download", True):
        raise FileNotFoundError(
            f"VAE checkpoint is missing and auto_download is false: {checkpoint}"
        )
    checksum = config.get("expected_sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in checksum)
    ):
        raise ValueError("VAE expected_sha256 must be a trusted 64-character hex checksum")
    config["expected_sha256"] = checksum.lower()
    if config.get("cache_latents", True):
        config["latent_cache_dir"] = _resolve_path(
            config.get("latent_cache_dir", "artifacts/latents"),
            base,
            "problem.encoder.latent_cache_dir",
        )
    return (image_shape[0] // 8, image_shape[1] // 8, 4)


def _validate_rhs(config: dict[str, Any], state_shape: tuple[int, ...], base: Path) -> None:
    rhs_type = config.get("type")
    compute_dtype = config.get("compute_dtype", "float32")
    if compute_dtype not in COMPUTE_DTYPES:
        raise ValueError(f"rhs.compute_dtype must be one of: {', '.join(sorted(COMPUTE_DTYPES))}")
    if rhs_type == "mlp":
        if len(state_shape) != 1 and not config.get("flatten", False):
            raise ValueError("MLP requires vector state or rhs.flatten=true")
        hidden = config.get("hidden_dims", [512, 512, 512])
        if not isinstance(hidden, list) or not hidden:
            raise ValueError("rhs.hidden_dims must be a non-empty literal list")
        for index, value in enumerate(hidden):
            _positive_integer(value, f"rhs.hidden_dims[{index}]")
        if config.get("activation", "silu") not in MLP_ACTIVATIONS:
            raise ValueError(
                f"rhs.activation must be one of: {', '.join(sorted(MLP_ACTIVATIONS))}"
            )
        time_embedding = _object(
            config.get("time_embedding", {}), "rhs.time_embedding"
        )
        if time_embedding.get("type", "sinusoidal") != "sinusoidal":
            raise ValueError("Only rhs.time_embedding.type='sinusoidal' is supported")
        _positive_integer(time_embedding.get("dim", 128), "rhs.time_embedding.dim")
        return
    if rhs_type not in {"unet", "sit"}:
        raise ValueError("rhs.type must be one of: mlp, unet, sit")
    if len(state_shape) != 3:
        raise ValueError(f"{rhs_type} requires a spatial state")
    if rhs_type == "unet":
        variant = config.get("variant", "small")
        if variant not in UNET_VARIANTS:
            raise ValueError(
                f"rhs.variant must be one of: {', '.join(sorted(UNET_VARIANTS))}"
            )
    if rhs_type == "sit":
        obsolete = {"source_dir", "source_auto_download"} & set(config)
        if obsolete:
            raise ValueError(
                "Packaged DiffuseNNX does not accept obsolete SiT source fields: "
                + ", ".join(sorted(obsolete))
            )
        if config.get("implementation", "diffuse_nnx") != "diffuse_nnx":
            raise ValueError("Only rhs.implementation='diffuse_nnx' is supported for SiT")
        variant = str(config.get("variant", "S")).upper()
        if variant not in SIT_VARIANTS:
            raise ValueError(f"SiT rhs.variant must be one of: {', '.join(sorted(SIT_VARIANTS))}")
        config["variant"] = variant
        patch_size = _positive_integer(config.get("patch_size", 2), "rhs.patch_size")
        if state_shape[0] % patch_size or state_shape[1] % patch_size:
            raise ValueError("SiT patch_size must divide both spatial dimensions")
        if config.get("class_conditioning", False):
            raise ValueError("Image benchmark SiT must be unconditional")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = _read_with_extends(path)
    keys = set(config)
    missing = REQUIRED_TOP_LEVEL - keys
    unknown = keys - REQUIRED_TOP_LEVEL - OPTIONAL_TOP_LEVEL
    if missing:
        raise ValueError(f"Missing top-level config keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {', '.join(sorted(unknown))}")
    config = copy.deepcopy(config)
    base = path.parent

    experiment = _object(config["experiment"], "experiment")
    if not isinstance(experiment.get("name"), str) or not experiment["name"]:
        raise ValueError("experiment.name must be a non-empty string")
    master_seed = experiment.get("seed", 0)
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise ValueError("experiment.seed must be an integer")
    experiment["seed"] = master_seed
    experiment["output_root"] = _resolve_path(
        experiment.get("output_root", "runs/image_benchmarks"),
        base,
        "experiment.output_root",
    )

    problem = _object(config["problem"], "problem")
    dataset = _object(problem.get("dataset"), "problem.dataset")
    if "hf_token" in dataset:
        raise ValueError(
            "Do not store problem.dataset.hf_token in JSON; configure HF_TOKEN in the environment"
        )
    spec = get_dataset_spec(dataset.get("name"))
    if "validation_size" in dataset:
        raise ValueError(
            "problem.dataset.validation_size is obsolete; official training sets are preserved"
        )
    if "train_size" in dataset:
        if spec.name != "ffhq64":
            raise ValueError("problem.dataset.train_size is only supported for ffhq64")
        _positive_integer(dataset["train_size"], "problem.dataset.train_size")
    resolution = spec.validate_resolution(dataset.get("resolution"))
    dataset["resolution"] = resolution
    dataset["cache_dir"] = _resolve_path(
        dataset.get("cache_dir", "data/huggingface"), base, "problem.dataset.cache_dir"
    )
    if spec.gated and not dataset.get("offline", False):
        token = dataset.get("hf_token") or os.environ.get("HF_TOKEN")
        if not token:
            raise PermissionError(
                f"Dataset {spec.hf_id} is gated; accept its terms and configure HF_TOKEN"
            )
    encoder = _object(problem.get("encoder"), "problem.encoder")
    horizontal_flip = dataset.get("horizontal_flip", False)
    _boolean(horizontal_flip, "problem.dataset.horizontal_flip")
    if encoder.get("type") != "none" and horizontal_flip:
        raise ValueError(
            "Cached latent training cannot apply problem.dataset.horizontal_flip; "
            "set it to false for AE/VAE runs"
        )
    image_shape = (resolution, resolution, spec.channels)
    state_shape = _validate_encoder(encoder, image_shape, base)
    _validate_rhs(_object(config["rhs"], "rhs"), state_shape, base)

    training = _object(config["training"], "training")
    _positive_integer(training.get("max_steps"), "training.max_steps")
    batch_size = _positive_integer(training.get("batch_size"), "training.batch_size")
    if "target_loader_epochs" in training:
        training["target_loader_epochs"] = _positive_integer(
            training["target_loader_epochs"], "training.target_loader_epochs"
        )
    for name in ("log_every", "checkpoint_every", "validation_every"):
        training[name] = _positive_integer(training.get(name, 1000), f"training.{name}")
    training["keep_checkpoints"] = _positive_integer(
        training.get("keep_checkpoints", 3), "training.keep_checkpoints"
    )

    methods = config["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty array")
    for index, method in enumerate(methods):
        method = _object(method, f"methods[{index}]")
        if method.get("name") not in METHOD_NAMES:
            raise ValueError(f"Unknown methods[{index}].name {method.get('name')!r}")
        method["n_restarts"] = _positive_integer(
            method.get("n_restarts", 1), f"methods[{index}].n_restarts"
        )
        kwargs = _object(method.get("kwargs", {}), f"methods[{index}].kwargs")
        if method["name"] in {"ngd", "anderson"}:
            if "step_size" not in kwargs:
                raise ValueError(f"methods[{index}].kwargs.step_size is required")
        elif "learning_rate" not in kwargs and "step_size" not in kwargs:
            raise ValueError(f"methods[{index}].kwargs.learning_rate is required")
        if method["name"] == "anderson" and not (
            "regularization_factor" in kwargs or "reg_factor" in kwargs
        ):
            raise ValueError(
                f"methods[{index}].kwargs.regularization_factor is required"
            )
        schedule = kwargs.get("stepsize_schedule", kwargs.get("stepsize_schedule_name"))
        if schedule not in {None, "schedule_exp"}:
            raise ValueError("Only stepsize_schedule='schedule_exp' is supported")

    evaluation = _object(config["evaluation"], "evaluation")
    evaluation.setdefault("split", "test")
    if evaluation["split"] not in {"validation", "test"}:
        raise ValueError("evaluation.split must be one of: validation, test")
    evaluation.setdefault("seed", master_seed + 1000)
    validation = _object(
        evaluation.setdefault("val_fm_loss", {"enabled": True}),
        "evaluation.val_fm_loss",
    )
    validation.setdefault("enabled", True)
    if _boolean(validation["enabled"], "evaluation.val_fm_loss.enabled") is not True:
        raise ValueError("Fixed evaluation.val_fm_loss must remain enabled")
    validation["num_samples"] = _positive_integer(
        validation.get("num_samples", 5000), "evaluation.val_fm_loss.num_samples"
    )
    validation["batch_size"] = _positive_integer(
        validation.get("batch_size", batch_size), "evaluation.val_fm_loss.batch_size"
    )
    fid = _object(evaluation.setdefault("fid", {"enabled": False}), "evaluation.fid")
    fid.setdefault("enabled", False)
    _boolean(fid["enabled"], "evaluation.fid.enabled")
    if fid["enabled"]:
        obsolete = {"source_dir", "source_auto_download"} & set(fid)
        if obsolete:
            raise ValueError(
                "Packaged DiffuseNNX does not accept obsolete FID source fields: "
                + ", ".join(sorted(obsolete))
            )
        _boolean(fid.get("auto_download", True), "evaluation.fid.auto_download")
        fid["num_samples_final"] = _integer_at_least(
            fid.get("num_samples_final", 50000),
            2,
            "evaluation.fid.num_samples_final",
        )
        fid["cache_dir"] = _resolve_path(
            fid.get("cache_dir", "artifacts/fid"), base, "evaluation.fid.cache_dir"
        )
        fid["weights_path"] = _resolve_path(
            fid.get(
                "weights_path",
                "artifacts/inception/inception_v3_weights_fid.pickle",
            ),
            base,
            "evaluation.fid.weights_path",
        )
        checksum = fid.get(
            "expected_sha256",
            "4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77",
        )
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in checksum)
        ):
            raise ValueError(
                "evaluation.fid.expected_sha256 must be a trusted 64-character hex checksum"
            )
        fid["expected_sha256"] = checksum.lower()
        fid["batch_size"] = _positive_integer(
            fid.get("batch_size", 64), "evaluation.fid.batch_size"
        )
    kid = _object(evaluation.setdefault("kid", {"enabled": False}), "evaluation.kid")
    kid.setdefault("enabled", False)
    _boolean(kid["enabled"], "evaluation.kid.enabled")
    if kid["enabled"] and not fid["enabled"]:
        raise ValueError("evaluation.kid.enabled requires evaluation.fid.enabled")
    if kid["enabled"]:
        kid["subsets"] = _positive_integer(
            kid.get("subsets", 100), "evaluation.kid.subsets"
        )
        kid["subset_size"] = _positive_integer(
            kid.get("subset_size", 1000), "evaluation.kid.subset_size"
        )
    sampling = _object(evaluation.setdefault("sampling", {}), "evaluation.sampling")
    sampling.setdefault("method", "heun")
    if sampling["method"] not in SAMPLING_METHODS:
        raise ValueError(
            f"evaluation.sampling.method must be one of: {', '.join(sorted(SAMPLING_METHODS))}"
        )
    _object(sampling.setdefault("kwargs", {}), "evaluation.sampling.kwargs")
    sampling["steps"] = _positive_integer(
        sampling.get("steps", 50), "evaluation.sampling.steps"
    )
    sampling["batch_size"] = _positive_integer(
        sampling.get("batch_size", batch_size), "evaluation.sampling.batch_size"
    )
    sample_metrics = _object(
        evaluation.setdefault("sample_metrics", {}), "evaluation.sample_metrics"
    )
    sample_metrics["num_samples"] = _integer_at_least(
        sample_metrics.get("num_samples", 1000),
        2,
        "evaluation.sample_metrics.num_samples",
    )
    sample_metrics["batch_size"] = _positive_integer(
        sample_metrics.get("batch_size", sampling["batch_size"]),
        "evaluation.sample_metrics.batch_size",
    )
    mmd = _object(
        sample_metrics.setdefault("mmd", {"enabled": False}),
        "evaluation.sample_metrics.mmd",
    )
    mmd.setdefault("enabled", False)
    _boolean(mmd["enabled"], "evaluation.sample_metrics.mmd.enabled")
    if mmd["enabled"]:
        choices = [name for name in ("bandwidths", "bw_multipliers") if name in mmd]
        if len(choices) != 1:
            raise ValueError(
                "Enabled MMD requires exactly one of evaluation.sample_metrics.mmd."
                "bandwidths or bw_multipliers"
            )
        mmd[choices[0]] = _positive_number_list(
            mmd[choices[0]], f"evaluation.sample_metrics.mmd.{choices[0]}"
        )
    sliced = _object(
        sample_metrics.setdefault("sliced_wasserstein", {"enabled": False}),
        "evaluation.sample_metrics.sliced_wasserstein",
    )
    sliced.setdefault("enabled", False)
    _boolean(
        sliced["enabled"], "evaluation.sample_metrics.sliced_wasserstein.enabled"
    )
    if sliced["enabled"]:
        sliced["num_projections"] = _positive_integer(
            sliced.get("num_projections", 100),
            "evaluation.sample_metrics.sliced_wasserstein.num_projections",
        )

    resources = _object(config["resources"], "resources")
    gpu_ids = resources.get("gpu_ids", [])
    if not isinstance(gpu_ids, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in gpu_ids
    ):
        raise ValueError("resources.gpu_ids must be an array of non-negative integers")
    resources["gpu_ids"] = gpu_ids
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("resources.gpu_ids must not contain duplicates")
    gpus_per_run = _positive_integer(
        resources.get("gpus_per_run", 1), "resources.gpus_per_run"
    )
    resources["gpus_per_run"] = gpus_per_run
    resources["max_concurrent_runs"] = _positive_integer(
        resources.get("max_concurrent_runs", 1), "resources.max_concurrent_runs"
    )
    if gpu_ids and gpus_per_run > len(gpu_ids):
        raise ValueError("resources.gpus_per_run exceeds available gpu_ids")
    if not gpu_ids and gpus_per_run != 1:
        raise ValueError("resources.gpus_per_run must be 1 when gpu_ids is empty")
    if batch_size % gpus_per_run:
        raise ValueError("training.batch_size must be divisible by resources.gpus_per_run")
    data_parallel = resources.setdefault("data_parallel", True)
    _boolean(data_parallel, "resources.data_parallel")
    if not data_parallel and gpus_per_run != 1:
        raise ValueError("resources.data_parallel=false requires gpus_per_run=1")
    worker_env = _object(resources.setdefault("worker_env", {}), "resources.worker_env")
    for key, value in worker_env.items():
        if (
            key == "CUDA_VISIBLE_DEVICES"
            or any(word in key.upper() for word in ("TOKEN", "PASSWORD", "SECRET"))
            or not isinstance(value, str)
        ):
            raise ValueError(
                "resources.worker_env values must be strings and cannot contain "
                "CUDA_VISIBLE_DEVICES or credential variables"
            )

    config.setdefault("plotting", {})
    config["resolved"] = {
        "config_path": str(path),
        "dataset_hf_id": spec.hf_id,
        "image_shape": list(image_shape),
        "state_shape": list(state_shape),
        "channels": spec.channels,
    }
    return config


def plan_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    master_seed = config["experiment"]["seed"]
    for method in config["methods"]:
        for restart_index in range(method["n_restarts"]):
            payload = {
                "master_seed": master_seed,
                "method": method,
                "restart_index": restart_index,
                "problem": config["problem"],
                "rhs": config["rhs"],
                "training": config["training"],
                "evaluation": config["evaluation"],
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:10]
            index = len(runs)
            run_id = f"run_{index:04d}_{method['name']}_r{restart_index:02d}_{digest}"
            runs.append(
                {
                    "run_index": index,
                    "run_id": run_id,
                    "method": copy.deepcopy(method),
                    "restart_index": restart_index,
                    "seed": _seed(master_seed, restart_index, "model"),
                    "rng_seeds": {
                        stream: _seed(master_seed, restart_index, stream)
                        for stream in RNG_STREAMS
                    },
                }
            )
    return runs


def gpu_groups(resources: dict[str, Any]) -> list[list[int]]:
    gpu_ids = resources["gpu_ids"]
    if not gpu_ids:
        return [[]]
    size = resources["gpus_per_run"]
    groups = [gpu_ids[start : start + size] for start in range(0, len(gpu_ids), size)]
    return [group for group in groups if len(group) == size][
        : resources["max_concurrent_runs"]
    ]
