import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from image_benchmarks.datasets.hf_loader import download_dataset, load_split
from image_benchmarks.datasets.manifest import DatasetManifest
from image_benchmarks.datasets.registry import DATASET_REGISTRY, get_dataset_spec
from image_benchmarks.datasets.transforms import (
    model_to_evaluation,
    transform_image,
)


class FakeSplit:
    def __init__(self, records, fingerprint):
        self.records = records
        self._fingerprint = fingerprint

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [record[item] for record in self.records]
        return self.records[item]


def fake_mnist():
    train = [
        {
            "image": np.full((7, 5), index * 10, dtype=np.uint8),
            "label": index % 10,
        }
        for index in range(12)
    ]
    test = [
        {"image": np.full((6, 8), 255 - index, dtype=np.uint8), "label": index}
        for index in range(4)
    ]
    return {
        "train": FakeSplit(train, "fake-train"),
        "test": FakeSplit(test, "fake-test"),
    }


def fake_loader_factory(dataset):
    def load_dataset(*args, **kwargs):
        del args, kwargs
        return dataset

    return load_dataset


def test_registry_contains_all_required_datasets():
    expected = {
        "mnist",
        "fashion_mnist",
        "cifar10",
        "flowers102",
        "afhq_cat",
        "lsun_church",
        "ffhq64",
        "imagenet64",
        "imagenet256",
    }
    assert set(DATASET_REGISTRY) == expected
    assert get_dataset_spec("cifar10").hf_id == "uoft-cs/cifar10"
    assert get_dataset_spec("ffhq64").filename_key is None


def test_manifest_and_splits_are_process_independent():
    dataset = fake_mnist()
    loader = fake_loader_factory(dataset)
    resolver = lambda *args: "fake-revision"
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        first = download_dataset(
            "mnist",
            first_dir,
            split_seed=17,
            load_dataset_fn=loader,
            revision_resolver=resolver,
        )
        second = download_dataset(
            "mnist",
            second_dir,
            split_seed=17,
            load_dataset_fn=loader,
            revision_resolver=resolver,
        )
        assert first.digest == second.digest
        for name in first.splits:
            first_indices = np.load(
                first.manifest_dir / first.splits[name].indices_file
            )
            second_indices = np.load(
                second.manifest_dir / second.splits[name].indices_file
            )
            np.testing.assert_array_equal(first_indices, second_indices)
        assert first.summary()["split_counts"] == {
            "train": 12,
            "validation": 4,
            "test": 4,
        }
        assert first.splits["validation"].source_split == "test"
        loaded = DatasetManifest.read(first.path)
        assert loaded.digest == first.digest


def test_official_test_alias_does_not_reduce_training_set():
    dataset = fake_mnist()
    loader = fake_loader_factory(dataset)
    with tempfile.TemporaryDirectory() as directory:
        first = download_dataset(
            "mnist",
            directory,
            split_seed=2,
            load_dataset_fn=loader,
            revision_resolver=lambda *args: "fake-revision",
        )
        second = download_dataset(
            "mnist",
            directory,
            split_seed=3,
            load_dataset_fn=loader,
            revision_resolver=lambda *args: "fake-revision",
        )
        assert first.manifest_dir != second.manifest_dir
        assert first.splits["train"].count == 12
        assert second.splits["train"].count == 12
        assert first.splits["validation"].count == 4
        assert second.splits["validation"].count == 4


def test_lazy_loader_returns_normalized_nhwc_batches():
    dataset = fake_mnist()
    loader = fake_loader_factory(dataset)
    with tempfile.TemporaryDirectory() as directory:
        manifest = download_dataset(
            "mnist",
            directory,
            resolution=10,
            load_dataset_fn=loader,
            revision_resolver=lambda *args: "fake-revision",
        )
        iterator = load_split(
            manifest,
            "validation",
            batch_size=2,
            seed=5,
            load_dataset_fn=loader,
        )
        batch = next(iter(iterator))
        assert batch["image"].shape == (2, 10, 10, 1)
        assert batch["image"].dtype == np.float32
        assert np.min(batch["image"]) >= -1.0
        assert np.max(batch["image"]) <= 1.0
        assert len(batch["id"]) == 2


def test_transform_round_trip_and_deterministic_flip():
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    first = transform_image(
        image,
        resolution=4,
        channels=3,
        horizontal_flip=True,
        rng=np.random.default_rng(9),
    )
    second = transform_image(
        image,
        resolution=4,
        channels=3,
        horizontal_flip=True,
        rng=np.random.default_rng(9),
    )
    np.testing.assert_array_equal(first, second)
    evaluation = model_to_evaluation(first)
    assert evaluation.shape == (4, 4, 3)
    assert evaluation.dtype == np.uint8


def test_gated_dataset_requires_credentials_before_download():
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict("os.environ", {}, clear=True):
            try:
                download_dataset(
                    "imagenet256",
                    directory,
                    hf_token=None,
                    load_dataset_fn=lambda *args, **kwargs: {},
                )
            except PermissionError as error:
                assert "HF_TOKEN" in str(error)
            else:
                raise AssertionError("Expected gated ImageNet credential validation")


def test_imagenet_keeps_full_train_and_official_validation_as_evaluation():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    dataset = {
        "train": FakeSplit([{"image": image, "label": 0}] * 10, "train"),
        "validation": FakeSplit([{"image": image, "label": 0}] * 4, "validation"),
    }
    with tempfile.TemporaryDirectory() as directory:
        manifest = download_dataset(
            "imagenet64",
            directory,
            load_dataset_fn=fake_loader_factory(dataset),
            revision_resolver=lambda *args: "fake-revision",
        )
        assert manifest.splits["validation"].source_split == "validation"
        assert manifest.splits["validation"].count == 4
        assert manifest.splits["test"].source_split == "validation"
        assert manifest.splits["train"].count == 10
        assert "fm_validation" not in manifest.splits


def test_ffhq_train_size_and_seed_are_manifest_identity():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    dataset = {"train": FakeSplit([{"image": image}] * 70000, "ffhq")}
    loader = fake_loader_factory(dataset)
    resolver = lambda *args: "fake-revision"
    with tempfile.TemporaryDirectory() as directory:
        default = download_dataset(
            "ffhq64", directory, load_dataset_fn=loader, revision_resolver=resolver
        )
        smaller = download_dataset(
            "ffhq64",
            directory,
            train_size=55000,
            load_dataset_fn=loader,
            revision_resolver=resolver,
        )
        reseeded = download_dataset(
            "ffhq64",
            directory,
            train_size=55000,
            split_seed=9,
            load_dataset_fn=loader,
            revision_resolver=resolver,
        )
        assert default.train_size == 60000
        assert default.summary()["split_counts"] == {
            "train": 60000,
            "validation": 10000,
            "test": 10000,
        }
        assert smaller.train_size == 55000
        assert smaller.splits["train"].count == 55000
        assert smaller.splits["test"].count == 15000
        assert smaller.manifest_dir != default.manifest_dir
        assert reseeded.manifest_dir != smaller.manifest_dir


def test_offline_placeholder_is_not_used_as_hf_revision():
    dataset = fake_mnist()
    calls = []

    def loader(*args, **kwargs):
        del args
        calls.append(kwargs)
        return dataset

    with tempfile.TemporaryDirectory() as directory:
        manifest = download_dataset(
            "mnist",
            directory,
            offline=True,
            load_dataset_fn=loader,
        )
        list(load_split(manifest, "test", 2, 0, offline=True, load_dataset_fn=loader))
        assert manifest.hf_revision == "cached-unresolved"
        assert calls[-1]["revision"] is None


def test_split_order_matches_across_independent_processes():
    program = (
        f"import json, sys; sys.path.insert(0, {str(ROOT / 'examples')!r}); "
        "from image_benchmarks.datasets.splits import stable_index_order; "
        "print(json.dumps(stable_index_order(100, 23).tolist()))"
    )
    first = subprocess.check_output([sys.executable, "-c", program], text=True)
    second = subprocess.check_output([sys.executable, "-c", program], text=True)
    assert first == second


def test_loader_rejects_changed_source_fingerprint():
    dataset = fake_mnist()
    with tempfile.TemporaryDirectory() as directory:
        manifest = download_dataset(
            "mnist",
            directory,
            load_dataset_fn=fake_loader_factory(dataset),
            revision_resolver=lambda *args: "unresolved",
        )
        changed = fake_mnist()
        changed["test"]._fingerprint = "different-test"
        try:
            load_split(
                manifest,
                "test",
                2,
                0,
                load_dataset_fn=fake_loader_factory(changed),
            )
        except ValueError as error:
            assert "fingerprint changed" in str(error)
        else:
            raise AssertionError("Expected changed source fingerprint rejection")


if __name__ == "__main__":
    test_registry_contains_all_required_datasets()
    test_manifest_and_splits_are_process_independent()
    test_official_test_alias_does_not_reduce_training_set()
    test_lazy_loader_returns_normalized_nhwc_batches()
    test_transform_round_trip_and_deterministic_flip()
    test_gated_dataset_requires_credentials_before_download()
    test_imagenet_keeps_full_train_and_official_validation_as_evaluation()
    test_ffhq_train_size_and_seed_are_manifest_identity()
    test_offline_placeholder_is_not_used_as_hf_revision()
    test_split_order_matches_across_independent_processes()
    test_loader_rejects_changed_source_fingerprint()
    print("Image dataset tests passed.")
