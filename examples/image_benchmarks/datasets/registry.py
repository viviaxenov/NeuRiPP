"""Dataset metadata for unconditional image benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


PROJECT_SPLIT_SEED = 20260811


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    image_key: str
    label_key: str | None
    default_resolution: int
    channels: int
    split_recipe: str
    preprocessing: str = "center_square"
    supported_resolutions: tuple[int, ...] | None = None
    gated: bool = False
    filename_key: str | None = None
    default_validation_size: int | float | None = None

    def validate_resolution(self, resolution: int | None) -> int:
        resolved = self.default_resolution if resolution is None else resolution
        if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
            raise ValueError("dataset resolution must be a positive integer")
        if self.supported_resolutions and resolved not in self.supported_resolutions:
            choices = ", ".join(str(value) for value in self.supported_resolutions)
            raise ValueError(
                f"Dataset {self.name!r} supports resolutions: {choices}; got {resolved}"
            )
        return resolved


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        hf_id="ylecun/mnist",
        image_key="image",
        label_key="label",
        default_resolution=28,
        channels=1,
        split_recipe="train_holdout_test",
        default_validation_size=5000,
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        hf_id="uoft-cs/cifar10",
        image_key="img",
        label_key="label",
        default_resolution=32,
        channels=3,
        split_recipe="train_holdout_test",
        default_validation_size=5000,
    ),
    "flowers102": DatasetSpec(
        name="flowers102",
        hf_id="pufanyi/flowers102",
        image_key="image",
        label_key="label",
        default_resolution=64,
        channels=3,
        split_recipe="provided_three_way",
        supported_resolutions=(64, 256),
    ),
    "afhq_cat": DatasetSpec(
        name="afhq_cat",
        hf_id="bitmind/AFHQ",
        image_key="image",
        label_key=None,
        default_resolution=256,
        channels=3,
        split_recipe="afhq_cat",
        supported_resolutions=(256, 512),
        filename_key="filename",
        default_validation_size=0.1,
    ),
    "lsun_church": DatasetSpec(
        name="lsun_church",
        hf_id="tglcourse/lsun_church_train",
        image_key="image",
        label_key=None,
        default_resolution=256,
        channels=3,
        split_recipe="train_holdout_test",
        supported_resolutions=(256,),
        default_validation_size=5000,
    ),
    "ffhq64": DatasetSpec(
        name="ffhq64",
        hf_id="Dmini/FFHQ-64x64",
        image_key="image",
        label_key=None,
        default_resolution=64,
        channels=3,
        split_recipe="ffhq_60_5_5",
        supported_resolutions=(64,),
    ),
    "imagenet64": DatasetSpec(
        name="imagenet64",
        hf_id="benjamin-paine/imagenet-1k-64x64",
        image_key="image",
        label_key="label",
        default_resolution=64,
        channels=3,
        split_recipe="imagenet",
        supported_resolutions=(64,),
        default_validation_size=5000,
    ),
    "imagenet256": DatasetSpec(
        name="imagenet256",
        hf_id="ILSVRC/imagenet-1k",
        image_key="image",
        label_key="label",
        default_resolution=256,
        channels=3,
        split_recipe="imagenet",
        supported_resolutions=(256,),
        gated=True,
        default_validation_size=5000,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as error:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown image dataset {name!r}; expected one of: {supported}"
        ) from error
