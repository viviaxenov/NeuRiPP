"""Deterministic image conversion and benchmark preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_pil(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = np.rint(array * 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def transform_image(
    image: Any,
    *,
    resolution: int,
    channels: int,
    crop: str = "center_square",
    horizontal_flip: bool = False,
    rng: np.random.Generator | None = None,
    output: str = "model",
) -> np.ndarray:
    """Transform one image to NHWC model or evaluation convention."""

    from PIL import Image

    pil_image = _to_pil(image)
    if channels == 1:
        pil_image = pil_image.convert("L")
    elif channels == 3:
        pil_image = pil_image.convert("RGB")
    else:
        raise ValueError(f"Only one or three image channels are supported; got {channels}")

    if crop == "center_square":
        width, height = pil_image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        pil_image = pil_image.crop((left, top, left + side, top + side))
    elif crop != "none":
        raise ValueError(f"Unsupported crop policy {crop!r}")

    pil_image = pil_image.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    if horizontal_flip:
        if rng is None:
            raise ValueError("rng is required for stochastic horizontal flipping")
        if bool(rng.random() < 0.5):
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    array = np.asarray(pil_image, dtype=np.uint8)
    if channels == 1:
        array = array[..., None]
    if output == "evaluation":
        return array
    if output != "model":
        raise ValueError("output must be 'model' or 'evaluation'")
    return array.astype(np.float32) / 127.5 - 1.0


def model_to_evaluation(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    return np.clip(np.rint((images + 1.0) * 127.5), 0, 255).astype(np.uint8)
