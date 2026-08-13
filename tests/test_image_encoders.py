import tempfile
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from image_benchmarks.assets.files import prepare_vae_checkpoint
from image_benchmarks.encoders.cache import (
    LatentCacheKey,
    LatentCacheWriter,
    open_latent_cache,
    sample_cached_stats,
    write_latent_cache,
)
from image_benchmarks.encoders.diffuse_vae import DiffuseVAEEncoder
from image_benchmarks.encoders.identity import IdentityEncoder
from image_benchmarks.encoders.project_ae import (
    ProjectAEEncoder,
)
from image_benchmarks.encoders.project_ae_training import build_autoencoder
from image_benchmarks.encoders.registry import encoder_state_shape


class FakeAE:
    def encode(self, images):
        return images.reshape(images.shape[0], -1)[:, :3]

    def decode(self, latent):
        values = jnp.pad(latent, ((0, 0), (0, 1)))
        return values.reshape(latent.shape[0], 2, 2, 1)


class FakeVAECore:
    def encoder(self, images, deterministic=True):
        del deterministic
        return images[:, ::8, ::8, :]

    def quant_conv(self, hidden):
        mean = jnp.concatenate((hidden, hidden[..., :1]), axis=-1)
        log_variance = jnp.zeros_like(mean)
        return jnp.concatenate((mean, log_variance), axis=-1)


class FakeVAEModel:
    def __init__(self):
        self.vae = FakeVAECore()

    def decode(self, latent, deterministic=True):
        del deterministic
        value = jnp.clip(jnp.mean(latent, axis=-1, keepdims=True), -1.0, 1.0)
        value = jnp.repeat(jnp.repeat(value, 8, axis=1), 8, axis=2)
        value = jnp.repeat(value, 3, axis=-1)
        return jnp.clip(jnp.rint((value + 1.0) * 127.5), 0, 255).astype(jnp.uint8)


def test_identity_round_trip():
    encoder = IdentityEncoder()
    images = np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 2, 2, 3)
    assert encoder.latent_shape((2, 2, 3)) == (2, 2, 3)
    np.testing.assert_array_equal(encoder.decode(encoder.encode(images)), images)


def test_project_ae_adapter_converts_image_convention():
    encoder = ProjectAEEncoder(FakeAE(), latent_dim=3)
    images = jnp.linspace(-1.0, 1.0, 8).reshape(2, 2, 2, 1)
    latent = encoder.encode(images)
    decoded = encoder.decode(latent)
    assert latent.shape == (2, 3)
    assert decoded.shape == images.shape
    assert encoder.latent_shape((2, 2, 1)) == (3,)
    assert float(decoded.min()) >= -1.0
    assert float(decoded.max()) <= 1.0


def test_project_ae_adapter_uses_existing_model_implementation():
    model = build_autoencoder((2, 2, 1), 3, rng_seed=2)
    encoder = ProjectAEEncoder(model, latent_dim=3)
    latent = encoder.encode(jnp.zeros((2, 2, 2, 1)))
    assert latent.shape == (2, 3)


def test_diffuse_vae_stats_scale_sampling_and_decode():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "vae.pkl"
        checkpoint.write_bytes(b"checkpoint")
        encoder = DiffuseVAEEncoder(
            FakeVAEModel(), checkpoint, sample_posterior=True
        )
        images = jnp.ones((2, 16, 16, 3))
        mean, std = encoder.encode_stats(images)
        assert mean.shape == (2, 2, 2, 4)
        np.testing.assert_allclose(std, encoder.scale_factor)
        first = encoder.sample_from_stats(mean, std, jax.random.key(3))
        second = encoder.sample_from_stats(mean, std, jax.random.key(3))
        np.testing.assert_array_equal(first, second)
        decoded = encoder.decode(mean)
        assert decoded.shape == (2, 16, 16, 3)
        assert decoded.dtype == jnp.float32
        assert encoder.latent_shape((16, 16, 3)) == (2, 2, 4)


def test_vae_asset_preparation_is_idempotent_and_hashed():
    calls = []

    def download(url, destination):
        calls.append(url)
        destination.write_bytes(b"vae-weights")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vae_trial1.pkl"
        first = prepare_vae_checkpoint(path, download_fn=download)
        second = prepare_vae_checkpoint(path, download_fn=download)
        assert first == second
        assert len(calls) == 1
        assert first["size_bytes"] == len(b"vae-weights")


def test_vae_asset_rejects_checksum_before_publication():
    def download(url, destination):
        del url
        destination.write_bytes(b"untrusted")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vae_trial1.pkl"
        try:
            prepare_vae_checkpoint(
                path,
                expected_sha256="0" * 64,
                download_fn=download,
            )
        except RuntimeError as error:
            assert "Failed to download" in str(error)
        else:
            raise AssertionError("Expected VAE checksum rejection")
        assert not path.exists()


def test_latent_cache_key_invalidation_and_round_trip():
    base = LatentCacheKey("revision-a", "train", 256, "center_square", "abc", 0.18215)
    changed = LatentCacheKey("revision-a", "train", 64, "center_square", "abc", 0.18215)
    assert base.digest != changed.digest
    changed_split = LatentCacheKey(
        "revision-a",
        "train",
        256,
        "center_square",
        "abc",
        0.18215,
        split_indices_sha256="different",
    )
    assert base.digest != changed_split.digest
    mean = np.zeros((3, 2, 2, 4), dtype=np.float32)
    std = np.ones_like(mean)
    with tempfile.TemporaryDirectory() as directory:
        cache = write_latent_cache(directory, base, mean, std, ["a", "b", "c"])
        reopened = open_latent_cache(directory, base)
        loaded_mean, loaded_std, identifiers = reopened.load()
        np.testing.assert_array_equal(loaded_mean, mean)
        np.testing.assert_array_equal(loaded_std, std)
        assert identifiers == ["a", "b", "c"]
        first = sample_cached_stats(loaded_mean, loaded_std, jax.random.key(9))
        second = sample_cached_stats(loaded_mean, loaded_std, jax.random.key(9))
        np.testing.assert_array_equal(first, second)
        assert cache.directory == reopened.directory


def test_encoder_registry_resolves_representation_shapes():
    assert encoder_state_shape({"type": "none"}, (32, 32, 3)) == (32, 32, 3)
    assert encoder_state_shape({"type": "ae", "latent_dim": 64}, (28, 28, 1)) == (64,)
    assert encoder_state_shape({"type": "vae"}, (256, 256, 3)) == (32, 32, 4)


def test_incremental_latent_cache_writer():
    key = LatentCacheKey("revision", "train", 256, "center_square", "hash", 0.18215)
    with tempfile.TemporaryDirectory() as directory:
        writer = LatentCacheWriter(
            directory, key, count=3, latent_shape=(2, 2, 4), dtype=np.float32
        )
        writer.write_batch(
            np.zeros((2, 2, 2, 4)),
            np.ones((2, 2, 2, 4)),
            ["first", "second"],
        )
        writer.write_batch(
            np.zeros((1, 2, 2, 4)), np.ones((1, 2, 2, 4)), ["third"]
        )
        cache = writer.finalize()
        mean, std, identifiers = cache.load(
            expected_identifiers=["first", "second", "third"],
            verify_checksums=True,
        )
        assert mean.shape == (3, 2, 2, 4)
        assert std.shape == mean.shape
        assert identifiers == ["first", "second", "third"]


def test_repeated_latent_cache_publication_keeps_valid_cache():
    key = LatentCacheKey("revision", "test", 64, "center_square", "hash", 0.18215)
    mean = np.zeros((2, 1, 1, 4), dtype=np.float32)
    std = np.ones_like(mean)
    with tempfile.TemporaryDirectory() as directory:
        first = write_latent_cache(directory, key, mean, std, ["a", "b"])
        second = write_latent_cache(directory, key, mean, std, ["a", "b"])
        assert first.directory == second.directory
        second.load(verify_checksums=True)


def test_conflicting_latent_cache_publication_is_rejected():
    key = LatentCacheKey("revision", "validation", 64, "center_square", "hash", 0.18215)
    mean = np.zeros((2, 1, 1, 4), dtype=np.float32)
    std = np.ones_like(mean)
    with tempfile.TemporaryDirectory() as directory:
        write_latent_cache(directory, key, mean, std, ["a", "b"])
        try:
            write_latent_cache(directory, key, mean + 1.0, std, ["a", "different"])
        except ValueError as error:
            assert "Conflicting latent cache" in str(error)
        else:
            raise AssertionError("Expected conflicting latent cache rejection")


if __name__ == "__main__":
    test_identity_round_trip()
    test_project_ae_adapter_converts_image_convention()
    test_project_ae_adapter_uses_existing_model_implementation()
    test_diffuse_vae_stats_scale_sampling_and_decode()
    test_vae_asset_preparation_is_idempotent_and_hashed()
    test_vae_asset_rejects_checksum_before_publication()
    test_latent_cache_key_invalidation_and_round_trip()
    test_encoder_registry_resolves_representation_shapes()
    test_incremental_latent_cache_writer()
    test_repeated_latent_cache_publication_keeps_valid_cache()
    test_conflicting_latent_cache_publication_is_rejected()
    print("Image encoder tests passed.")
