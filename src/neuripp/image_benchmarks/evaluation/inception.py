"""Thin adapter around DiffuseNNX's pinned FID Inception implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

from neuripp.image_benchmarks.assets.diffuse_nnx import prepare_diffuse_nnx_source


_IMPORT_LOCK = threading.Lock()


def _import_inception_without_torch(source_dir: Path):
    """Load upstream Inception while avoiding unrelated Torch loader imports."""

    source_dir = prepare_diffuse_nnx_source(source_dir, auto_download=False)
    module_name = f"_neuripp_diffuse_inception_{source_dir.name}"
    with _IMPORT_LOCK:
        if module_name in sys.modules:
            return sys.modules[module_name]
        saved = {name: sys.modules.get(name) for name in ("eval", "eval.utils")}
        eval_package = ModuleType("eval")
        eval_package.__path__ = [str(source_dir / "eval")]
        utility_stub = ModuleType("eval.utils")

        def get(mapping, key):
            return None if mapping is None or key not in mapping else mapping[key]

        utility_stub.get = get
        utility_stub.__file__ = str(source_dir / "eval" / "utils.py")
        eval_package.utils = utility_stub
        sys.modules["eval"] = eval_package
        sys.modules["eval.utils"] = utility_stub
        path = source_dir / "eval" / "inception.py"
        specification = importlib.util.spec_from_file_location(module_name, path)
        if specification is None or specification.loader is None:
            raise ImportError(f"Cannot load DiffuseNNX Inception from {path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        finally:
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
    module_path = Path(module.__file__).resolve()
    if source_dir not in module_path.parents:
        raise ImportError(f"DiffuseNNX Inception loaded outside pinned source: {module_path}")
    return module


class DiffuseInceptionFeatures:
    """Single-process feature extractor backed by DiffuseNNX InceptionV3."""

    feature_dim = 2048
    provenance = "diffuse_nnx_inception_v3_fid_023afd2"

    def __init__(self, source_dir: str | Path):
        module = _import_inception_without_torch(Path(source_dir).expanduser().resolve())
        self.detector = module.InceptionV3(pretrained=True)
        self.variables = self.detector.init(
            jax.random.key(0), jnp.ones((1, 299, 299, 3), dtype=jnp.float32)
        )

        def apply_detector(images):
            images = images.astype(jnp.float32) / 127.5 - 1.0
            images = jax.image.resize(
                images, (images.shape[0], 299, 299, 3), method="bilinear"
            )
            return self.detector.apply(self.variables, images, train=False).reshape(
                images.shape[0], -1
            )

        self._apply = jax.jit(apply_detector)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim != 4 or images.dtype != np.uint8:
            raise ValueError("Inception images must be uint8 NHWC batches")
        if images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        if images.shape[-1] != 3:
            raise ValueError("Inception images must have one or three channels")
        return np.asarray(self._apply(jnp.asarray(images)))
