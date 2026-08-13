"""Asset hashing and atomic download helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Callable
from urllib.request import urlopen


VAE_CHECKPOINT_URL = "gs://will-data/stats/vae_trial1.pkl"
INCEPTION_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/viviaxenov/diffuse_nnx/"
    "023afd23c7b62a8cdb00e840b36a4ab8fc970bba/"
    "eval/inception_v3_weights_fid.pickle"
)
INCEPTION_WEIGHTS_SHA256 = (
    "4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77"
)


def sha256_path(path: str | Path) -> str:
    """Hash a file or a checkpoint directory in stable relative-path order."""

    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _download_gcs(url: str, destination: Path) -> None:
    try:
        from google.cloud import storage
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "VAE download requires google-cloud-storage and application-default credentials"
        ) from error
    prefix = "gs://"
    if not url.startswith(prefix) or "/" not in url[len(prefix) :]:
        raise ValueError(f"Invalid Google Cloud Storage URL: {url}")
    bucket_name, blob_name = url[len(prefix) :].split("/", 1)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(destination)


def _download_http(url: str, destination: Path) -> None:
    """Stream an HTTPS asset to a temporary destination."""

    with urlopen(url, timeout=120) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def prepare_inception_weights(
    destination: str | Path,
    *,
    auto_download: bool = True,
    expected_sha256: str = INCEPTION_WEIGHTS_SHA256,
    download_fn: Callable[[str, Path], None] | None = None,
) -> dict[str, str | int]:
    """Prepare and verify the external DiffuseNNX Inception FID pickle."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists() and not destination.is_file():
        raise ValueError(f"Inception weights path is not a regular file: {destination}")
    if destination.is_file() and destination.stat().st_size == 0:
        raise ValueError(f"Inception weights file is empty: {destination}")
    if not destination.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"Inception weights do not exist: {destination}. Enable auto_download "
                "or prepare the verified asset before evaluation."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            (download_fn or _download_http)(INCEPTION_WEIGHTS_URL, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("download produced an empty Inception weights file")
            downloaded_checksum = sha256_path(temporary)
            if downloaded_checksum != expected_sha256:
                raise ValueError(
                    "Downloaded Inception weights checksum mismatch: "
                    f"expected {expected_sha256}, got {downloaded_checksum}"
                )
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        except Exception as error:
            raise RuntimeError(
                f"Failed to download DiffuseNNX Inception weights to {destination}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)
    checksum = sha256_path(destination)
    if checksum != expected_sha256:
        raise ValueError(
            f"Inception weights checksum mismatch: expected {expected_sha256}, got {checksum}"
        )
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": checksum,
    }


def prepare_vae_checkpoint(
    destination: str | Path,
    *,
    auto_download: bool = True,
    expected_sha256: str | None = None,
    download_fn: Callable[[str, Path], None] | None = None,
) -> dict[str, str | int]:
    """Prepare the pinned DiffuseNNX VAE checkpoint idempotently."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists() and not destination.is_file():
        raise ValueError(f"VAE checkpoint path is not a regular file: {destination}")
    if destination.is_file() and destination.stat().st_size == 0:
        raise ValueError(f"VAE checkpoint is empty: {destination}")
    if not destination.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"VAE checkpoint does not exist: {destination}. Enable auto_download "
                "or prepare the asset before training."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            (download_fn or _download_gcs)(VAE_CHECKPOINT_URL, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("download produced an empty checkpoint")
            downloaded_checksum = sha256_path(temporary)
            if expected_sha256 and downloaded_checksum != expected_sha256:
                raise ValueError(
                    "Downloaded VAE checkpoint checksum mismatch: "
                    f"expected {expected_sha256}, got {downloaded_checksum}"
                )
            try:
                os.link(temporary, destination)
            except FileExistsError:
                # A concurrent preparer published first; validate it below.
                pass
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download DiffuseNNX VAE checkpoint to {destination}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)
    checksum = sha256_path(destination)
    if expected_sha256 and checksum != expected_sha256:
        raise ValueError(
            f"VAE checkpoint checksum mismatch: expected {expected_sha256}, got {checksum}"
        )
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": checksum,
    }
