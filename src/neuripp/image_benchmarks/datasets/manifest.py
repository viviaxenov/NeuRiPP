"""Serializable dataset and split manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitManifest:
    source_split: str
    count: int
    indices_file: str
    indices_sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    hf_id: str
    hf_revision: str
    image_key: str
    label_key: str | None
    filename_key: str | None
    resolution: int
    channels: int
    crop: str
    normalization: str
    split_seed: int
    source_fingerprints: dict[str, str]
    splits: dict[str, SplitManifest]
    manifest_dir: Path

    @property
    def path(self) -> Path:
        return self.manifest_dir / "manifest.json"

    def to_dict(self, *, include_directory: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["splits"] = {
            name: asdict(split) for name, split in self.splits.items()
        }
        if include_directory:
            payload["manifest_dir"] = str(self.manifest_dir)
        else:
            payload.pop("manifest_dir", None)
        return payload

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_directory=False), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def write(self) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @classmethod
    def read(cls, path: str | Path) -> "DatasetManifest":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["manifest_dir"] = Path(payload.get("manifest_dir", path.parent))
        payload["splits"] = {
            name: SplitManifest(**split) for name, split in payload["splits"].items()
        }
        return cls(**payload)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "hf_revision": self.hf_revision,
            "resolution": self.resolution,
            "channels": self.channels,
            "normalization": self.normalization,
            "split_seed": self.split_seed,
            "split_counts": {
                name: split.count for name, split in self.splits.items()
            },
            "digest": self.digest,
        }
