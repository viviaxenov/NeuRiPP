"""Pinned DiffuseNNX source preparation and import."""

from __future__ import annotations

from importlib import import_module
import errno
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType


DIFFUSE_NNX_REPOSITORY = "https://github.com/willisma/diffuse_nnx.git"
DIFFUSE_NNX_COMMIT = "023afd23c7b62a8cdb00e840b36a4ab8fc970bba"


def _git_output(source_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_dir), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def prepare_diffuse_nnx_source(
    destination: str | Path, *, auto_download: bool = True
) -> Path:
    """Clone and verify the exact DiffuseNNX source revision when needed."""

    destination = Path(destination).expanduser().resolve()
    if not destination.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"DiffuseNNX source is missing at {destination}; run asset preparation "
                "with auto_download enabled."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        # git clone requires a nonexistent destination.
        temporary.rmdir()
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    DIFFUSE_NNX_REPOSITORY,
                    str(temporary),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(temporary),
                    "checkout",
                    "--detach",
                    DIFFUSE_NNX_COMMIT,
                ],
                check=True,
            )
            try:
                temporary.rename(destination)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                shutil.rmtree(temporary)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    try:
        revision = _git_output(destination, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(f"Invalid DiffuseNNX source checkout: {destination}") from error
    if revision != DIFFUSE_NNX_COMMIT:
        raise ValueError(
            f"DiffuseNNX source at {destination} is revision {revision}; "
            f"expected {DIFFUSE_NNX_COMMIT}"
        )
    dirty = _git_output(destination, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(
            f"DiffuseNNX source at {destination} has tracked modifications; "
            "use a clean pinned checkout"
        )
    return destination


def import_diffuse_module(module_name: str, source_dir: str | Path) -> ModuleType:
    """Import a module from the verified source fallback.

    The upstream wheel omits ``networks`` and ``eval``. Workers are isolated
    processes, so adding the verified checkout for these upstream absolute
    imports cannot leak between benchmark runs.
    """

    source_dir = prepare_diffuse_nnx_source(source_dir, auto_download=False)
    source_text = str(source_dir)
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if source_dir not in existing_path.parents:
            prefix = module_name.split(".", 1)[0]
            for name in [key for key in sys.modules if key == prefix or key.startswith(prefix + ".")]:
                del sys.modules[name]
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = import_module(module_name)
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if source_dir not in module_path.parents:
        raise ImportError(
            f"Imported {module_name} from {module_path}, outside pinned source {source_dir}"
        )
    return module
