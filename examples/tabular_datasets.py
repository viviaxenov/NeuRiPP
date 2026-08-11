"""Download and load the tabular density-estimation benchmarks used by MAF."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
from time import perf_counter
import urllib.request

import h5py
import numpy as np
import pandas as pd


MAF_ARCHIVE_URL = (
    "https://zenodo.org/api/records/1161203/files/data.tar.gz/content"
)
MAF_ARCHIVE_MD5 = "9b9c9b0375315ad270eba4ce80c093ab"
MAF_ARCHIVE_DOI = "10.5281/zenodo.1161203"
EXPECTED_DIMS = {
    "power": 6,
    "gas": 8,
    "hepmass": 21,
    "miniboone": 43,
    "bsds300": 63,
}
DATASET_FILES = {
    "power": ("power/data.npy",),
    "gas": ("gas/ethylene_CO.pickle",),
    "hepmass": ("hepmass/1000_train.csv", "hepmass/1000_test.csv"),
    "miniboone": ("miniboone/data.npy",),
    "bsds300": ("BSDS300/BSDS300.hdf5",),
}


@dataclass(frozen=True)
class TabularDataset:
    name: str
    train: np.ndarray
    test: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    download_wall_time: float
    preprocessing_wall_time: float
    data_dir: Path


def default_data_dir() -> Path:
    if value := os.environ.get("NEURIPP_DATA_DIR"):
        return Path(value).expanduser()
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "neuripp" / "maf"


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(data_dir: Path) -> tuple[Path, float]:
    archive_path = data_dir / "data.tar.gz"
    if archive_path.is_file() and _file_md5(archive_path) == MAF_ARCHIVE_MD5:
        return archive_path, 0.0

    data_dir.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(f".gz.part.{os.getpid()}")
    partial_path.unlink(missing_ok=True)
    start = perf_counter()
    try:
        with urllib.request.urlopen(MAF_ARCHIVE_URL, timeout=120) as response, partial_path.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        checksum = _file_md5(partial_path)
        if checksum != MAF_ARCHIVE_MD5:
            raise RuntimeError(
                f"MAF archive checksum mismatch: expected {MAF_ARCHIVE_MD5}, got {checksum}"
            )
        partial_path.replace(archive_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return archive_path, perf_counter() - start


def _find_archive_member(archive: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    normalized_suffix = suffix.lower().replace("\\", "/")
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and member.name.lower().replace("\\", "/").endswith(normalized_suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one archive member ending in {suffix!r}, found {len(matches)}"
        )
    return matches[0]


def ensure_dataset_files(dataset: str, data_dir: Path | None = None) -> tuple[Path, float]:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unsupported tabular dataset: {dataset}")
    data_dir = default_data_dir() if data_dir is None else Path(data_dir).expanduser()
    extracted_root = data_dir / "extracted"
    expected_paths = tuple(extracted_root / relative for relative in DATASET_FILES[dataset])
    if all(path.is_file() for path in expected_paths):
        return extracted_root, 0.0

    archive_path, download_wall_time = _download_archive(data_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        for relative, destination in zip(DATASET_FILES[dataset], expected_paths, strict=True):
            if destination.is_file():
                continue
            member = _find_archive_member(archive, relative)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {member.name!r} from MAF archive")
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + f".part.{os.getpid()}")
            with source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            partial.replace(destination)
    return extracted_root, download_wall_time


def _split_train_validation_test(data: np.ndarray):
    n_test = int(0.1 * len(data))
    test = data[-n_test:]
    train_and_validation = data[:-n_test]
    n_validation = int(0.1 * len(train_and_validation))
    return train_and_validation[:-n_validation], test


def _load_power(root: Path):
    data = np.load(root / DATASET_FILES["power"][0]).copy()
    rng = np.random.RandomState(42)
    rng.shuffle(data)
    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    noise = np.hstack(
        (
            0.001 * rng.rand(len(data), 1),
            0.01 * rng.rand(len(data), 1),
            rng.rand(len(data), 3),
            np.zeros((len(data), 1)),
        )
    )
    return _split_train_validation_test(data + noise)


def _load_gas(root: Path):
    data = pd.read_pickle(root / DATASET_FILES["gas"][0]).drop(
        columns=["Meth", "Eth", "Time"]
    )
    correlation_counts = (data.corr() > 0.98).sum(axis=1).to_numpy()
    while np.any(correlation_counts > 1):
        data = data.drop(columns=data.columns[np.flatnonzero(correlation_counts > 1)[0]])
        correlation_counts = (data.corr() > 0.98).sum(axis=1).to_numpy()
    return _split_train_validation_test(data.to_numpy())


def _load_hepmass(root: Path):
    train_frame = pd.read_csv(root / DATASET_FILES["hepmass"][0], index_col=False)
    test_frame = pd.read_csv(root / DATASET_FILES["hepmass"][1], index_col=False)
    train_frame = train_frame[train_frame.iloc[:, 0] == 1].iloc[:, 1:]
    test_frame = test_frame[test_frame.iloc[:, 0] == 1].iloc[:, 1:-1]
    train = train_frame.to_numpy()
    test = test_frame.to_numpy()

    remove = []
    for index, feature in enumerate(train.T):
        first_value_count = sorted(Counter(feature).items())[0][1]
        if first_value_count > 5:
            remove.append(index)
    keep = np.asarray([index for index in range(train.shape[1]) if index not in remove])
    train = train[:, keep]
    test = test[:, keep]
    n_validation = int(0.1 * len(train))
    return train[:-n_validation], test


def _load_miniboone(root: Path):
    data = np.load(root / DATASET_FILES["miniboone"][0])
    return _split_train_validation_test(data)


def _load_bsds300(root: Path):
    with h5py.File(root / DATASET_FILES["bsds300"][0], "r") as file:
        return np.asarray(file["train"]), np.asarray(file["test"])


LOADERS = {
    "power": _load_power,
    "gas": _load_gas,
    "hepmass": _load_hepmass,
    "miniboone": _load_miniboone,
    "bsds300": _load_bsds300,
}


def normalize_with_train_moments(train: np.ndarray, test: np.ndarray):
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.finfo(np.float32).eps)
    return (train - mean) / std, (test - mean) / std, mean, std


def load_tabular_dataset(dataset: str, data_dir: Path | None = None) -> TabularDataset:
    dataset = dataset.lower()
    root, download_wall_time = ensure_dataset_files(dataset, data_dir)
    start = perf_counter()
    train, test = LOADERS[dataset](root)
    train, test, mean, std = normalize_with_train_moments(train, test)

    expected_dim = EXPECTED_DIMS[dataset]
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != expected_dim:
        raise RuntimeError(
            f"Unexpected {dataset} shapes: train={train.shape}, test={test.shape}; "
            f"expected feature dimension {expected_dim}"
        )
    if test.shape[1] != expected_dim or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError(f"{dataset} contains invalid or non-finite values")

    resolved_data_dir = default_data_dir() if data_dir is None else Path(data_dir).expanduser()
    return TabularDataset(
        name=dataset,
        train=train,
        test=test,
        train_mean=mean,
        train_std=std,
        download_wall_time=download_wall_time,
        preprocessing_wall_time=perf_counter() - start,
        data_dir=resolved_data_dir,
    )
