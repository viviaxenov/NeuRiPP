"""Metadata and availability checks for the packaged DiffuseNNX integration."""

from __future__ import annotations

from importlib import import_module, metadata
import json
from pathlib import Path
from types import ModuleType


DIFFUSE_NNX_REPOSITORY = "https://github.com/viviaxenov/diffuse_nnx.git"
DIFFUSE_NNX_COMMIT = "da5f2b79497722931d279b012c90bec61050466b"
DIFFUSE_NNX_VERSION = "0.2.0.dev0"


def _normalized_repository(url: str) -> str:
    """Normalize accepted HTTPS/Git spelling without weakening repository identity."""

    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    elif url.startswith("ssh://git@github.com/"):
        url = "https://github.com/" + url.removeprefix("ssh://git@github.com/")
    return url.removesuffix(".git").rstrip("/")


def _verified_distribution() -> tuple[str, Path]:
    """Return the exact installed version and expected import-package root."""

    try:
        distribution = metadata.distribution("diffuse-nnx")
    except metadata.PackageNotFoundError as error:
        raise ModuleNotFoundError(
            "DiffuseNNX image components require NeuRiPP's "
            "'image-benchmarks-diffuse' optional dependencies"
        ) from error
    if distribution.version != DIFFUSE_NNX_VERSION:
        raise RuntimeError(
            f"Unsupported diffuse-nnx version {distribution.version}; "
            f"expected {DIFFUSE_NNX_VERSION} "
            f"from commit {DIFFUSE_NNX_COMMIT}"
        )
    direct_url = distribution.read_text("direct_url.json")
    try:
        provenance = json.loads(direct_url or "")
    except (json.JSONDecodeError, AttributeError) as error:
        raise RuntimeError("diffuse-nnx installation provenance is invalid") from error
    vcs_info = provenance.get("vcs_info", {})
    commit = vcs_info.get("commit_id")
    if commit is not None:
        if vcs_info.get("vcs") != "git":
            raise RuntimeError("diffuse-nnx provenance must use Git")
        if _normalized_repository(provenance.get("url", "")) != _normalized_repository(
            DIFFUSE_NNX_REPOSITORY
        ):
            raise RuntimeError(
                "diffuse-nnx provenance repository does not match the pinned fork"
            )
    if "dir_info" in provenance:
        raise RuntimeError(
            "editable diffuse-nnx installations are unsupported; install the exact "
            "VCS dependency through NeuRiPP[image-benchmarks-diffuse]"
        )
    package_root = Path(distribution.locate_file("diffuse_nnx")).resolve()
    if commit != DIFFUSE_NNX_COMMIT:
        raise RuntimeError(
            f"Unsupported diffuse-nnx commit {commit}; expected {DIFFUSE_NNX_COMMIT}"
        )
    if not (package_root / "__init__.py").is_file():
        raise RuntimeError(
            f"Verified diffuse-nnx distribution has no package at {package_root}"
        )
    return distribution.version, package_root.resolve()


def require_diffuse_nnx() -> str:
    """Return the installed version or explain which extra provides it."""

    return _verified_distribution()[0]


def import_diffuse_module(module_name: str) -> ModuleType:
    """Import a canonical module from the verified installed distribution."""

    _, expected_root = _verified_distribution()
    if not module_name.startswith("diffuse_nnx."):
        raise ValueError("DiffuseNNX imports must use the canonical diffuse_nnx namespace")
    package = import_module("diffuse_nnx")
    module = import_module(module_name)
    package_root = Path(package.__file__).resolve().parent
    module_path = Path(module.__file__).resolve()
    if package_root != expected_root or package_root not in module_path.parents:
        raise ImportError(
            f"Imported {module_name} from {module_path} under {package_root}, outside "
            f"verified diffuse-nnx distribution package {expected_root}"
        )
    return module
