"""Metadata and availability checks for the packaged DiffuseNNX integration."""

from __future__ import annotations

from importlib import import_module, metadata
import json
from pathlib import Path
import subprocess
from types import ModuleType
from urllib.parse import unquote, urlparse


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
    try:
        provenance = json.loads(direct_url or "")
    except (json.JSONDecodeError, AttributeError) as error:
        raise RuntimeError("diffuse-nnx installation provenance is invalid") from error
    commit = provenance.get("vcs_info", {}).get("commit_id")
    if commit is None and "dir_info" in provenance:
        parsed = urlparse(provenance.get("url", ""))
        if parsed.scheme != "file":
            raise RuntimeError("editable diffuse-nnx provenance is not a local Git checkout")
        checkout = Path(unquote(parsed.path)).resolve()
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "editable diffuse-nnx provenance is not a valid Git checkout"
            ) from error
    if commit != DIFFUSE_NNX_COMMIT:
        raise RuntimeError(
            f"Unsupported diffuse-nnx commit {commit}; expected {DIFFUSE_NNX_COMMIT}"
        )
    return version


def import_diffuse_module(module_name: str) -> ModuleType:
    """Import a canonical module from the verified installed distribution."""

    require_diffuse_nnx()
    if not module_name.startswith("diffuse_nnx."):
        raise ValueError("DiffuseNNX imports must use the canonical diffuse_nnx namespace")
    package = import_module("diffuse_nnx")
    module = import_module(module_name)
    package_root = Path(package.__file__).resolve().parent
    module_path = Path(module.__file__).resolve()
    if package_root not in module_path.parents:
        raise ImportError(
            f"Imported {module_name} from {module_path}, outside installed "
            f"diffuse_nnx package {package_root}"
        )
    return module
