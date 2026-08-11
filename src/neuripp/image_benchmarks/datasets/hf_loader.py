"""Hugging Face download, manifest, and lazy split iteration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from neuripp.image_benchmarks.datasets.manifest import DatasetManifest, SplitManifest
from neuripp.image_benchmarks.datasets.registry import (
    PROJECT_SPLIT_SEED,
    DatasetSpec,
    get_dataset_spec,
)
from neuripp.image_benchmarks.datasets.splits import (
    read_indices,
    split_fixed_counts,
    split_holdout,
    write_indices,
)
from neuripp.image_benchmarks.datasets.transforms import transform_image


def _default_load_dataset(*args, offline: bool = False, **kwargs):
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=offline)
    return load_dataset(*args, download_config=download_config, **kwargs)


def _resolve_revision(
    hf_id: str, revision: str | None, token: str | None, offline: bool
) -> str:
    if offline:
        return revision or "cached-unresolved"
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(hf_id, revision=revision, token=token).sha
    except Exception:
        return revision or "unresolved"


def _split_names(dataset: Mapping[str, Any]) -> list[str]:
    return list(dataset.keys())


def _column(split: Any, key: str) -> list[Any]:
    try:
        values = split[key]
        return list(values)
    except (KeyError, TypeError, ValueError):
        try:
            return [split[index][key] for index in range(len(split))]
        except (KeyError, TypeError, ValueError):
            raise KeyError(key) from None


def _identifiers(split: Any, source_name: str, filename_key: str | None) -> list[str]:
    if filename_key is not None:
        try:
            return [str(value) for value in _column(split, filename_key)]
        except KeyError:
            pass
    return [f"{source_name}:{index}" for index in range(len(split))]


def _source_fingerprints(dataset: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: str(getattr(split, "_fingerprint", f"count:{len(split)}"))
        for name, split in dataset.items()
    }


def _require_split(dataset: Mapping[str, Any], *names: str) -> str:
    for name in names:
        if name in dataset:
            return name
    available = ", ".join(_split_names(dataset))
    raise ValueError(f"Required dataset split {names} not found; available: {available}")


def _holdout_selection(
    dataset: Mapping[str, Any],
    spec: DatasetSpec,
    validation_size: int | float,
    seed: int,
) -> dict[str, tuple[str, np.ndarray]]:
    train_name = _require_split(dataset, "train")
    test_name = _require_split(dataset, "test")
    train_source = dataset[train_name]
    indices = np.arange(len(train_source), dtype=np.int64)
    train, validation = split_holdout(
        indices,
        (
            _identifiers(train_source, train_name, spec.filename_key)
            if spec.filename_key
            else None
        ),
        validation_size,
        seed,
    )
    return {
        "train": (train_name, train),
        "validation": (train_name, validation),
        "test": (test_name, np.arange(len(dataset[test_name]), dtype=np.int64)),
    }


def _afhq_selection(
    dataset: Mapping[str, Any],
    spec: DatasetSpec,
    seed: int,
    validation_size: int | float,
) -> dict[str, tuple[str, np.ndarray]]:
    train_candidates: tuple[str, np.ndarray] | None = None
    test_candidates: tuple[str, np.ndarray] | None = None
    for source_name, source in dataset.items():
        filenames = [value.replace("\\", "/") for value in _identifiers(source, source_name, spec.filename_key)]
        source_train = np.asarray(
            [index for index, value in enumerate(filenames) if "train/cat/" in value],
            dtype=np.int64,
        )
        source_test = np.asarray(
            [index for index, value in enumerate(filenames) if "test/cat/" in value],
            dtype=np.int64,
        )
        if source_train.size:
            if train_candidates is not None:
                raise ValueError("AFHQ cat train files span multiple HF splits")
            train_candidates = (source_name, source_train)
        if source_test.size:
            if test_candidates is not None:
                raise ValueError("AFHQ cat test files span multiple HF splits")
            test_candidates = (source_name, source_test)
    if train_candidates is None or test_candidates is None:
        raise ValueError(
            "AFHQ cat filtering found no train/cat or test/cat filenames; "
            "verify the pinned dataset revision"
        )
    train_source_name, candidates = train_candidates
    source = dataset[train_source_name]
    all_ids = _identifiers(source, train_source_name, spec.filename_key)
    candidate_ids = [all_ids[index] for index in candidates]
    train, validation = split_holdout(
        candidates, candidate_ids, validation_size, seed
    )
    return {
        "train": (train_source_name, train),
        "validation": (train_source_name, validation),
        "test": test_candidates,
    }


def _build_selections(
    dataset: Mapping[str, Any],
    spec: DatasetSpec,
    seed: int,
    validation_size: int | float | None,
) -> dict[str, tuple[str, np.ndarray]]:
    if spec.split_recipe == "train_holdout_test":
        size = spec.default_validation_size if validation_size is None else validation_size
        if size is None:
            raise ValueError(f"No validation size configured for {spec.name}")
        return _holdout_selection(dataset, spec, size, seed)

    if spec.split_recipe == "provided_three_way":
        train = _require_split(dataset, "train")
        validation = _require_split(dataset, "validation", "valid")
        test = _require_split(dataset, "test")
        return {
            "train": (train, np.arange(len(dataset[train]), dtype=np.int64)),
            "validation": (
                validation,
                np.arange(len(dataset[validation]), dtype=np.int64),
            ),
            "test": (test, np.arange(len(dataset[test]), dtype=np.int64)),
        }

    if spec.split_recipe == "afhq_cat":
        size = spec.default_validation_size if validation_size is None else validation_size
        if size is None:
            raise ValueError("No validation size configured for AFHQ cat")
        return _afhq_selection(dataset, spec, seed, size)

    if spec.split_recipe == "ffhq_60_5_5":
        source_name = "train" if "train" in dataset else _split_names(dataset)[0]
        source = dataset[source_name]
        if len(source) != 70000:
            raise ValueError(f"FFHQ-64 expected 70000 examples; got {len(source)}")
        indices = np.arange(len(source), dtype=np.int64)
        train, validation, test = split_fixed_counts(
            indices,
            (
                _identifiers(source, source_name, spec.filename_key)
                if spec.filename_key
                else None
            ),
            (60000, 5000, 5000),
            seed,
        )
        return {
            "train": (source_name, train),
            "validation": (source_name, validation),
            "test": (source_name, test),
        }

    if spec.split_recipe == "imagenet":
        train_name = _require_split(dataset, "train")
        reference_name = _require_split(dataset, "validation", "val")
        train_source = dataset[train_name]
        size = spec.default_validation_size if validation_size is None else validation_size
        if size:
            train, fm_validation = split_holdout(
                np.arange(len(train_source), dtype=np.int64),
                None,
                size,
                seed,
            )
        else:
            train = np.arange(len(train_source), dtype=np.int64)
            fm_validation = None
        reference_indices = np.arange(len(dataset[reference_name]), dtype=np.int64)
        selections = {
            "train": (train_name, train),
            "validation": (reference_name, reference_indices),
            "test": (reference_name, reference_indices.copy()),
        }
        if fm_validation is not None:
            selections["fm_validation"] = (train_name, fm_validation)
        return selections

    raise ValueError(f"Unknown split recipe {spec.split_recipe!r}")


def _manifest_directory(
    cache_dir: Path,
    spec: DatasetSpec,
    revision: str,
    resolution: int,
    crop: str,
    split_seed: int,
    validation_size: int | float | None,
) -> Path:
    key = json.dumps(
        {
            "name": spec.name,
            "revision": revision,
            "resolution": resolution,
            "crop": crop,
            "split_seed": split_seed,
            "validation_size": validation_size,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / "neuripp_manifests" / spec.name / digest


def download_dataset(
    spec: DatasetSpec | str,
    cache_dir: str | Path,
    *,
    hf_token: str | None = None,
    revision: str | None = None,
    resolution: int | None = None,
    crop: str | None = None,
    split_seed: int = PROJECT_SPLIT_SEED,
    validation_size: int | float | None = None,
    offline: bool = False,
    load_dataset_fn: Callable[..., Mapping[str, Any]] | None = None,
    revision_resolver: Callable[[str, str | None, str | None, bool], str] | None = None,
) -> DatasetManifest:
    """Resolve an HF dataset and persist deterministic logical split indices."""

    if isinstance(spec, str):
        spec = get_dataset_spec(spec)
    resolution = spec.validate_resolution(resolution)
    crop = spec.preprocessing if crop is None else crop
    token = hf_token or os.environ.get("HF_TOKEN")
    if spec.gated and not token and not offline:
        raise PermissionError(
            f"Dataset {spec.hf_id!r} is gated. Accept its Hugging Face terms and "
            "set HF_TOKEN or provide hf_token."
        )
    resolver = revision_resolver or _resolve_revision
    resolved_revision = resolver(spec.hf_id, revision, token, offline)
    loader = load_dataset_fn or _default_load_dataset
    try:
        requested_splits = (
            ["train", "validation"] if spec.split_recipe == "imagenet" else None
        )
        dataset = loader(
            spec.hf_id,
            cache_dir=str(Path(cache_dir)),
            token=token,
            revision=(
                resolved_revision
                if resolved_revision not in {"unresolved", "cached-unresolved"}
                else revision
            ),
            offline=offline,
            **({"split": requested_splits} if requested_splits else {}),
        )
        if requested_splits and not isinstance(dataset, Mapping):
            dataset = dict(zip(requested_splits, dataset, strict=True))
    except Exception as error:
        if spec.gated:
            raise PermissionError(
                f"Unable to access gated dataset {spec.hf_id!r}. Confirm that the "
                "account associated with HF_TOKEN accepted the dataset terms."
            ) from error
        if offline:
            raise RuntimeError(
                f"Dataset {spec.hf_id!r} is unavailable in the offline cache at {cache_dir}"
            ) from error
        raise

    selections = _build_selections(
        dataset, spec, split_seed, validation_size=validation_size
    )
    manifest_dir = _manifest_directory(
        Path(cache_dir),
        spec,
        resolved_revision,
        resolution,
        crop,
        split_seed,
        validation_size,
    )
    split_manifests: dict[str, SplitManifest] = {}
    for logical_name, (source_name, indices) in selections.items():
        filename = f"{logical_name}_indices.npy"
        indices_path = manifest_dir / filename
        write_indices(indices_path, indices)
        indices_sha256 = hashlib.sha256(indices_path.read_bytes()).hexdigest()
        split_manifests[logical_name] = SplitManifest(
            source_split=source_name,
            count=int(len(indices)),
            indices_file=filename,
            indices_sha256=indices_sha256,
        )
    manifest = DatasetManifest(
        name=spec.name,
        hf_id=spec.hf_id,
        hf_revision=resolved_revision,
        image_key=spec.image_key,
        label_key=spec.label_key,
        filename_key=spec.filename_key,
        resolution=resolution,
        channels=spec.channels,
        crop=crop,
        normalization="float32_nhwc_minus1_plus1",
        split_seed=split_seed,
        source_fingerprints=_source_fingerprints(dataset),
        splits=split_manifests,
        manifest_dir=manifest_dir,
    )
    manifest.write()
    return manifest


class ImageSplitIterator:
    """Lazy deterministic iterator over one logical manifest split."""

    def __init__(
        self,
        source: Any,
        indices: np.ndarray,
        manifest: DatasetManifest,
        split: str,
        batch_size: int,
        seed: int,
        *,
        augmentation_seed: int | None,
        shuffle: bool,
        horizontal_flip: bool,
        drop_last: bool,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.manifest = manifest
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.augmentation_seed = seed if augmentation_seed is None else augmentation_seed
        self.shuffle = shuffle
        self.horizontal_flip = horizontal_flip
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = self.epoch
        self.epoch += 1
        return self.iter_epoch(epoch)

    def iter_epoch(
        self, epoch: int, *, start_batch: int = 0
    ) -> Iterator[dict[str, Any]]:
        """Iterate a specific epoch, enabling exact checkpoint resume."""

        epoch_seed = self.seed + epoch
        order_rng = np.random.default_rng(epoch_seed)
        order = np.arange(len(self.indices))
        if self.shuffle:
            order_rng.shuffle(order)
        stop = len(order)
        if self.drop_last:
            stop = stop - stop % self.batch_size
        for start in range(start_batch * self.batch_size, stop, self.batch_size):
            positions = order[start : start + self.batch_size]
            if self.drop_last and len(positions) != self.batch_size:
                continue
            sample_indices = self.indices[positions]
            images = []
            labels = []
            identifiers = []
            for position, source_index in enumerate(sample_indices):
                record = self.source[int(source_index)]
                augmentation_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [self.augmentation_seed + epoch, int(source_index), position]
                    )
                )
                images.append(
                    transform_image(
                        record[self.manifest.image_key],
                        resolution=self.manifest.resolution,
                        channels=self.manifest.channels,
                        crop=self.manifest.crop,
                        horizontal_flip=self.horizontal_flip,
                        rng=augmentation_rng,
                    )
                )
                if self.manifest.label_key and self.manifest.label_key in record:
                    labels.append(record[self.manifest.label_key])
                if self.manifest.filename_key and self.manifest.filename_key in record:
                    identifiers.append(str(record[self.manifest.filename_key]))
                else:
                    identifiers.append(
                        f"{self.manifest.splits[self.split].source_split}:{source_index}"
                    )
            batch: dict[str, Any] = {
                "image": np.stack(images, axis=0),
                "id": identifiers,
                "index": sample_indices.copy(),
            }
            if labels:
                batch["label"] = np.asarray(labels)
            yield batch


def load_split(
    manifest: DatasetManifest | str | Path,
    split: str,
    batch_size: int,
    seed: int,
    *,
    shuffle: bool | None = None,
    augmentation_seed: int | None = None,
    horizontal_flip: bool = False,
    drop_last: bool = False,
    offline: bool = False,
    hf_token: str | None = None,
    load_dataset_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> ImageSplitIterator:
    if not isinstance(manifest, DatasetManifest):
        manifest = DatasetManifest.read(manifest)
    if split not in manifest.splits:
        available = ", ".join(sorted(manifest.splits))
        raise ValueError(f"Unknown split {split!r}; available splits: {available}")
    if split != "train" and horizontal_flip:
        raise ValueError("horizontal_flip is only valid for the training split")
    split_manifest = manifest.splits[split]
    loader = load_dataset_fn or _default_load_dataset
    loaded = loader(
        manifest.hf_id,
        cache_dir=str(manifest.manifest_dir.parents[2]),
        token=hf_token or os.environ.get("HF_TOKEN"),
        revision=(
            manifest.hf_revision
            if manifest.hf_revision not in {"unresolved", "cached-unresolved"}
            else None
        ),
        offline=offline,
        split=split_manifest.source_split,
    )
    source = (
        loaded[split_manifest.source_split]
        if isinstance(loaded, Mapping)
        else loaded
    )
    current_fingerprint = str(
        getattr(source, "_fingerprint", f"count:{len(source)}")
    )
    expected_fingerprint = manifest.source_fingerprints[split_manifest.source_split]
    if current_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Dataset source fingerprint changed for {manifest.name}/"
            f"{split_manifest.source_split}: expected {expected_fingerprint}, "
            f"got {current_fingerprint}. Rebuild the manifest."
        )
    indices_path = manifest.manifest_dir / split_manifest.indices_file
    actual_checksum = hashlib.sha256(indices_path.read_bytes()).hexdigest()
    if actual_checksum != split_manifest.indices_sha256:
        raise ValueError(
            f"Split index checksum mismatch for {manifest.name}/{split}: "
            f"expected {split_manifest.indices_sha256}, got {actual_checksum}"
        )
    indices = read_indices(indices_path)
    return ImageSplitIterator(
        source,
        indices,
        manifest,
        split,
        batch_size,
        seed,
        augmentation_seed=augmentation_seed,
        shuffle=split == "train" if shuffle is None else shuffle,
        horizontal_flip=horizontal_flip,
        drop_last=drop_last,
    )
