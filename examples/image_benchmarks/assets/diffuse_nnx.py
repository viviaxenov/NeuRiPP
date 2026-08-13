"""Metadata and availability checks for the packaged DiffuseNNX integration."""

from __future__ import annotations

from importlib import import_module, metadata
import json
from types import ModuleType


DIFFUSE_NNX_REPOSITORY = "https://github.com/viviaxenov/diffuse_nnx.git"
DIFFUSE_NNX_COMMIT = "da5f2b79497722931d279b012c90bec61050466b"
DIFFUSE_NNX_VERSION = "0.2.0.dev0"


def require_diffuse_nnx() -> str:
    """Return the installed version or explain which extra provides it."""

    try:
        version = metadata.version("diffuse-nnx")
    except metadata.PackageNotFoundError as error:
        raise ModuleNotFoundError(
            "DiffuseNNX image components require NeuRiPP's "
            "'image-benchmarks-diffuse' optional dependencies"
        ) from error
    if version != DIFFUSE_NNX_VERSION:
        raise RuntimeError(
            f"Unsupported diffuse-nnx version {version}; expected {DIFFUSE_NNX_VERSION} "
            f"from commit {DIFFUSE_NNX_COMMIT}"
        )
    direct_url = metadata.distribution("diffuse-nnx").read_text("direct_url.json")
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if commit is not None and commit != DIFFUSE_NNX_COMMIT:
            raise RuntimeError(
                f"Unsupported diffuse-nnx commit {commit}; expected {DIFFUSE_NNX_COMMIT}"
            )
    return version


def import_diffuse_module(module_name: str) -> ModuleType:
    """Import a canonical module from the verified installed distribution."""

    require_diffuse_nnx()
    if not module_name.startswith("diffuse_nnx."):
        raise ValueError("DiffuseNNX imports must use the canonical diffuse_nnx namespace")
    return import_module(module_name)
