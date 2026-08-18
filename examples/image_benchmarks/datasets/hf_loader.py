"""Hugging Face download, manifest, and lazy split iteration."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from image_benchmarks.datasets.manifest import DatasetManifest, SplitManifest
from image_benchmarks.datasets.registry import (
    PROJECT_SPLIT_SEED,
    DatasetSpec,
    get_dataset_spec,
)
from image_benchmarks.datasets.splits import (
    read_indices,
    stable_index_order,
    write_indices,
)
from image_benchmarks.datasets.transforms import transform_image


def _default_load_dataset(*args, offline: bool = False, **kwargs):
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=offline)
    return load_dataset(*args, download_config=download_config, **kwargs)


def _load_zip_imagefolder(spec: DatasetSpec, *args, offline: bool = False, **kwargs):
    """Load a dataset that ships as a bare image archive (zip) on the Hub.

    The Hub's automatic parquet conversion of such repositories can declare an
    incompatible ``ClassLabel`` feature, which makes the default ``datasets``
    loader fail on every hub-based path (plain load, feature override, and
    hub-backed ``imagefolder``).  This loader downloads the archive directly,
    extracts it once into the cache, and loads it as a local ``imagefolder``
    dataset so the rest of the harness sees a regular split mapping.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    cache_dir = Path(kwargs.pop("cache_dir", None) or ".")
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    split = kwargs.pop("split", None)
    if kwargs:
        raise TypeError(f"Unexpected loader kwargs: {sorted(kwargs)}")
    if spec.archive_file is None:
        raise ValueError(f"loader {spec.loader!r} requires spec.archive_file")

    zip_path = hf_hub_download(
        spec.hf_id,
        spec.archive_file,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        token=token,
        local_files_only=offline,
    )

    raw_dir = cache_dir / "raw" / spec.name
    if not (raw_dir.is_dir() and any(raw_dir.iterdir())):
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = raw_dir.with_name(f"{raw_dir.name}.tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)
        try:
            os.replace(tmp_dir, raw_dir)
        except OSError:
            # A concurrent process published the extraction first.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not (raw_dir.is_dir() and any(raw_dir.iterdir())):
                raise

    return load_dataset(
        "imagefolder",
        data_dir=str(raw_dir),
        cache_dir=str(cache_dir),
        **{
            key: value
            for key, value in {"split": split, "revision": revision, "token": token}.items()
            if value is not None
        },
    )


def _build_loader(spec: DatasetSpec, override: Callable[..., Mapping[str, Any]] | None):
    """Pick the dataset loader, honoring an explicit override first."""
    if override is not None:
        return override
    if spec.loader == "default":
        return _default_load_dataset
    if spec.loader == "zip_imagefolder":
        return functools.partial(_load_zip_imagefolder, spec)
    raise ValueError(f"Unknown dataset loader {spec.loader!r} for {spec.name!r}")


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


def _official_train_test_selection(
    dataset: Mapping[str, Any],
) -> dict[str, tuple[str, np.ndarray]]:
    train_name = _require_split(dataset, "train")
    test_name = _require_split(dataset, "test")
    train = np.arange(len(dataset[train_name]), dtype=np.int64)
    test = np.arange(len(dataset[test_name]), dtype=np.int64)
    return {
        "train": (train_name, train),
        "validation": (test_name, test.copy()),
        "test": (test_name, test),
    }


def _afhq_selection(
    dataset: Mapping[str, Any],
    spec: DatasetSpec,
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
    return {
        "train": (train_source_name, candidates),
        "validation": (test_candidates[0], test_candidates[1].copy()),
        "test": test_candidates,
    }


def _build_selections(
    dataset: Mapping[str, Any],
    spec: DatasetSpec,
    seed: int,
    train_size: int | None,
) -> dict[str, tuple[str, np.ndarray]]:
    if spec.split_recipe == "official_train_test":
        return _official_train_test_selection(dataset)

    if spec.split_recipe == "provided_three_way":
        train = _require_split(dataset, "train")
        test = _require_split(dataset, "test")
        test_indices = np.arange(len(dataset[test]), dtype=np.int64)
        return {
            "train": (train, np.arange(len(dataset[train]), dtype=np.int64)),
            "validation": (test, test_indices.copy()),
            "test": (test, test_indices),
        }

    if spec.split_recipe == "afhq_cat":
        return _afhq_selection(dataset, spec)

    if spec.split_recipe == "full_train_reference":
        train_name = _require_split(dataset, "train")
        reference_name = _require_split(dataset, "test", "validation", "val")
        reference = np.arange(len(dataset[reference_name]), dtype=np.int64)
        return {
            "train": (train_name, np.arange(len(dataset[train_name]), dtype=np.int64)),
            "validation": (reference_name, reference.copy()),
            "test": (reference_name, reference),
        }

    if spec.split_recipe == "ffhq_random":
        source_name = "train" if "train" in dataset else _split_names(dataset)[0]
        source = dataset[source_name]
        if len(source) != 70000:
            raise ValueError(f"FFHQ-64 expected 70000 examples; got {len(source)}")
        indices = np.arange(len(source), dtype=np.int64)
        resolved_train_size = spec.default_train_size if train_size is None else train_size
        if not isinstance(resolved_train_size, int) or isinstance(resolved_train_size, bool):
            raise ValueError("FFHQ train_size must be an integer")
        if not 1 <= resolved_train_size < len(indices):
            raise ValueError(
                f"FFHQ train_size must be between 1 and {len(indices) - 1}"
            )
        order = stable_index_order(len(indices), seed)
        train = indices[order[:resolved_train_size]]
        test = indices[order[resolved_train_size:]]
        return {
            "train": (source_name, train),
            "validation": (source_name, test.copy()),
            "test": (source_name, test),
        }

    if spec.split_recipe == "imagenet":
        train_name = _require_split(dataset, "train")
        reference_name = _require_split(dataset, "validation", "val")
        train_source = dataset[train_name]
        train = np.arange(len(train_source), dtype=np.int64)
        reference_indices = np.arange(len(dataset[reference_name]), dtype=np.int64)
        selections = {
            "train": (train_name, train),
            "validation": (reference_name, reference_indices),
            "test": (reference_name, reference_indices.copy()),
        }
        return selections

    raise ValueError(f"Unknown split recipe {spec.split_recipe!r}")


def _manifest_directory(
    cache_dir: Path,
    spec: DatasetSpec,
    revision: str,
    resolution: int,
    crop: str,
    split_seed: int,
    train_size: int | None,
) -> Path:
    key = json.dumps(
        {
            "name": spec.name,
            "revision": revision,
            "resolution": resolution,
            "crop": crop,
            "split_seed": split_seed,
            "train_size": train_size,
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
    train_size: int | None = None,
    offline: bool = False,
    load_dataset_fn: Callable[..., Mapping[str, Any]] | None = None,
    revision_resolver: Callable[[str, str | None, str | None, bool], str] | None = None,
) -> DatasetManifest:
    """Resolve an HF dataset and persist deterministic logical split indices."""

    if isinstance(spec, str):
        spec = get_dataset_spec(spec)
    if train_size is not None and spec.split_recipe != "ffhq_random":
        raise ValueError("train_size is only configurable for FFHQ-64")
    if spec.split_recipe == "ffhq_random" and train_size is None:
        train_size = spec.default_train_size
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
    loader = _build_loader(spec, load_dataset_fn)
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
        dataset, spec, split_seed, train_size=train_size
    )
    manifest_dir = _manifest_directory(
        Path(cache_dir),
        spec,
        resolved_revision,
        resolution,
        crop,
        split_seed,
        train_size,
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
        train_size=train_size,
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
    loader = _build_loader(get_dataset_spec(manifest.name), load_dataset_fn)
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
