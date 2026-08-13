import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from image_benchmarks.config import gpu_groups, load_config, plan_runs


def base_config(tmpdir):
    return {
        "experiment": {"name": "test", "seed": 7, "output_root": "runs"},
        "problem": {
            "dataset": {
                "name": "cifar10",
                "resolution": 32,
                "cache_dir": "hf",
            },
            "encoder": {"type": "none"},
        },
        "rhs": {
            "type": "unet",
            "variant": "small",
            "channel_mult": [1, 2, 2, 2],
            "attention_resolutions": [16],
        },
        "training": {"max_steps": 2, "batch_size": 4},
        "methods": [
            {"name": "adamw", "n_restarts": 2, "kwargs": {"learning_rate": 1e-3}},
            {"name": "ngd", "n_restarts": 2, "kwargs": {"step_size": 1e-3}},
        ],
        "evaluation": {
            "val_fm_loss": {"enabled": True, "num_samples": 4, "batch_size": 2},
            "fid": {"enabled": False},
            "kid": {"enabled": False},
            "sampling": {"method": "euler", "steps": 1, "batch_size": 2},
        },
        "resources": {
            "gpu_ids": [0, 1, 2, 3],
            "gpus_per_run": 2,
            "max_concurrent_runs": 2,
            "worker_env": {},
        },
    }


def write_config(directory, payload):
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_lists_are_literal_and_only_methods_restarts_expand():
    with tempfile.TemporaryDirectory() as directory:
        config = load_config(write_config(directory, base_config(directory)))
        runs = plan_runs(config)
        assert len(runs) == 4
        assert config["rhs"]["channel_mult"] == [1, 2, 2, 2]
        assert config["rhs"]["attention_resolutions"] == [16]
        assert runs[0]["rng_seeds"] == runs[2]["rng_seeds"]
        assert runs[1]["rng_seeds"] == runs[3]["rng_seeds"]


def test_resource_groups_reserve_disjoint_devices():
    groups = gpu_groups(
        {"gpu_ids": list(range(7)), "gpus_per_run": 2, "max_concurrent_runs": 3}
    )
    assert groups == [[0, 1], [2, 3], [4, 5]]


def test_invalid_encoder_rhs_pair_fails_preflight():
    with tempfile.TemporaryDirectory() as directory:
        payload = base_config(directory)
        payload["problem"]["dataset"]["name"] = "mnist"
        payload["problem"]["dataset"]["resolution"] = 28
        payload["problem"]["encoder"] = {
            "type": "ae",
            "latent_dim": 8,
            "checkpoint": "missing",
            "train_if_missing": True,
        }
        payload["rhs"] = {"type": "unet", "variant": "small"}
        try:
            load_config(write_config(directory, payload))
        except ValueError as error:
            assert "spatial state" in str(error)
        else:
            raise AssertionError("Expected AE-vector/U-Net preflight failure")


def test_config_inheritance_deep_merges_objects_and_replaces_arrays():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        parent = base_config(directory)
        (directory / "parent.json").write_text(json.dumps(parent), encoding="utf-8")
        child = {
            "extends": "parent.json",
            "experiment": {"name": "child"},
            "rhs": {"channel_mult": [1, 3]},
            "methods": [
                {"name": "adamw", "kwargs": {"learning_rate": 2e-3}}
            ],
        }
        path = directory / "child.json"
        path.write_text(json.dumps(child), encoding="utf-8")
        config = load_config(path)
        assert config["experiment"]["name"] == "child"
        assert config["experiment"]["seed"] == 7
        assert config["rhs"]["channel_mult"] == [1, 3]
        assert len(plan_runs(config)) == 1


def test_runtime_choices_fail_during_jax_free_preflight():
    cases = [
        ("rhs", {"type": "unet", "variant": "typo"}, "rhs.variant"),
        (
            "sampling",
            {"method": "bogus", "steps": 1, "batch_size": 2},
            "sampling.method",
        ),
    ]
    for target, replacement, expected in cases:
        with tempfile.TemporaryDirectory() as directory:
            payload = base_config(directory)
            if target == "rhs":
                payload["rhs"] = replacement
            else:
                payload["evaluation"][target] = replacement
            try:
                load_config(write_config(directory, payload))
            except ValueError as error:
                assert expected in str(error)
            else:
                raise AssertionError(f"Expected preflight failure for {target}")


def test_all_required_presets_resolve_and_plan():
    required = {
        "mnist_mlp.json",
        "mnist_ae32_mlp.json",
        "mnist_ae64_mlp.json",
        "cifar10_unet_small.json",
        "cifar10_unet.json",
        "flowers64_unet.json",
        "flowers256_vae_unet.json",
        "flowers256_vae_sit_s2.json",
        "ffhq64_unet.json",
        "afhqcat256_vae_unet.json",
        "lsun256_vae_unet.json",
        "lsun256_vae_sit_s2.json",
        "imagenet64_unet.json",
        "imagenet64_sit_s2.json",
        "imagenet256_vae_sit_s2.json",
        "imagenet256_vae_sit_b2.json",
    }
    root = ROOT / "examples" / "image_benchmarks" / "configs"
    assert required <= {path.name for path in root.glob("*.json")}
    previous = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = "structural-validation-token"
    try:
        for name in sorted(required):
            runs = plan_runs(load_config(root / name))
            assert runs
            assert len({run["run_id"] for run in runs}) == len(runs)
    finally:
        if previous is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = previous


def test_runtime_defaults_are_persisted_for_run_planning():
    with tempfile.TemporaryDirectory() as directory:
        payload = base_config(directory)
        payload["experiment"].pop("seed")
        payload["resources"].pop("gpu_ids")
        payload["resources"].pop("gpus_per_run")
        config = load_config(write_config(directory, payload))
        assert config["experiment"]["seed"] == 0
        assert config["resources"]["gpu_ids"] == []
        assert config["resources"]["gpus_per_run"] == 1
        assert plan_runs(config)
        assert gpu_groups(config["resources"]) == [[]]


def test_fid_requires_at_least_two_fake_samples():
    with tempfile.TemporaryDirectory() as directory:
        payload = base_config(directory)
        payload["evaluation"]["fid"] = {
            "enabled": True,
            "num_samples_final": 1,
        }
        try:
            load_config(write_config(directory, payload))
        except ValueError as error:
            assert "at least 2" in str(error)
        else:
            raise AssertionError("Expected one-sample FID preflight failure")


def test_obsolete_diffuse_source_checkout_fields_are_rejected():
    cases = (
        ("rhs", "source_dir"),
        ("fid", "source_auto_download"),
    )
    for target, field in cases:
        with tempfile.TemporaryDirectory() as directory:
            payload = base_config(directory)
            if target == "rhs":
                payload["rhs"] = {
                    "type": "sit",
                    "variant": "S",
                    "source_dir": "legacy-checkout",
                }
            else:
                payload["evaluation"]["fid"] = {
                    "enabled": True,
                    "num_samples_final": 2,
                    "source_auto_download": True,
                }
            try:
                load_config(write_config(directory, payload))
            except ValueError as error:
                assert field in str(error)
            else:
                raise AssertionError(f"Expected obsolete {field} rejection")


if __name__ == "__main__":
    test_lists_are_literal_and_only_methods_restarts_expand()
    test_resource_groups_reserve_disjoint_devices()
    test_invalid_encoder_rhs_pair_fails_preflight()
    test_config_inheritance_deep_merges_objects_and_replaces_arrays()
    test_runtime_choices_fail_during_jax_free_preflight()
    test_all_required_presets_resolve_and_plan()
    test_runtime_defaults_are_persisted_for_run_planning()
    test_fid_requires_at_least_two_fake_samples()
    test_obsolete_diffuse_source_checkout_fields_are_rejected()
    print("Image config tests passed.")
