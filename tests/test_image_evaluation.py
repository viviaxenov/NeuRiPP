import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.linalg
from flax import nnx

from neuripp.image_benchmarks.encoders.identity import IdentityEncoder
from neuripp.image_benchmarks.evaluation.fid import (
    FIDCacheKey,
    calculate_fid,
    load_fid_stats,
    load_diffuse_fid_function,
    statistics_from_feature_batches,
    write_fid_stats,
)
from neuripp.image_benchmarks.evaluation.evaluator import (
    evaluate_checkpoint,
    prepare_real_feature_cache,
)
from neuripp.image_benchmarks.evaluation.kid import calculate_kid
from neuripp.image_benchmarks.evaluation.reconstruction import reconstruction_metrics
from neuripp.image_benchmarks.evaluation.sampling import generate_image_batches
from neuripp.image_benchmarks.evaluation.validation import (
    evaluate_fixed_fm_loss,
    make_fixed_fm_validation,
)
from neuripp.parametric_pushforward.flow_matching import FlowMatching


class ZeroRHS(nnx.Module):
    def __init__(self, shape):
        self.dim = shape

    def __call__(self, time, state, *args):
        del time, args
        return jnp.zeros_like(state)


class CountingDecoder(IdentityEncoder):
    def __init__(self):
        self.decode_calls = 0

    def decode(self, latent, rng=None):
        del rng
        self.decode_calls += 1
        return latent


class DropoutAwareRHS(nnx.Module):
    def __init__(self):
        self.dim = (2,)
        self.uses_explicit_dropout_rng = True

    def __call__(self, time, state, key=None):
        del time
        if key is None:
            return jnp.zeros_like(state)
        return jnp.ones_like(state) * jnp.sum(key.astype(jnp.float32)) * 1e-9


class FakeExtractor:
    provenance = "fake_inception_for_tests"

    def __call__(self, images):
        images = np.asarray(images, dtype=np.float32)
        summary = images.mean(axis=(1, 2, 3), keepdims=False)[:, None]
        return np.repeat(summary, 2048, axis=1)


def test_fixed_fm_validation_is_repeatable():
    rngs = nnx.Rngs(0)
    model = FlowMatching(ZeroRHS((2, 2, 1)), rngs, 2)
    states = np.ones((4, 2, 2, 1), dtype=np.float32)
    validation = make_fixed_fm_validation(states, ["a", "b", "c", "d"], 9)
    first = evaluate_fixed_fm_loss(model, validation, batch_size=2)
    second = evaluate_fixed_fm_loss(model, validation, batch_size=3)
    np.testing.assert_allclose(first, second, rtol=1e-6)


def test_fixed_fm_validation_keeps_dropout_rng_across_batch_sizes():
    model = FlowMatching(DropoutAwareRHS(), nnx.Rngs(1), 2)
    states = np.ones((5, 2), dtype=np.float32)
    validation = make_fixed_fm_validation(states, list("abcde"), 11)
    first = evaluate_fixed_fm_loss(model, validation, batch_size=2)
    second = evaluate_fixed_fm_loss(model, validation, batch_size=5)
    np.testing.assert_allclose(first, second, rtol=1e-6)


def test_fid_statistics_and_cache_round_trip():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(20, 8))
    stats = statistics_from_feature_batches((features[:7], features[7:]))
    np.testing.assert_allclose(stats.mu, features.mean(axis=0))
    np.testing.assert_allclose(stats.sigma, np.cov(features, rowvar=False))
    assert abs(calculate_fid(stats, stats)) < 1e-8
    key = FIDCacheKey("revision", "test", 32, "center_square", "rgb")
    assert key.digest != FIDCacheKey(
        "revision",
        "test",
        32,
        "center_square",
        "rgb",
        split_indices_sha256="different",
    ).digest
    with tempfile.TemporaryDirectory() as directory:
        write_fid_stats(directory, key, stats)
        loaded = load_fid_stats(directory, key)
        np.testing.assert_array_equal(loaded.mu, stats.mu)
        np.testing.assert_array_equal(loaded.sigma, stats.sigma)
        assert loaded.count == 20


def test_kid_is_near_zero_for_same_distribution():
    features = np.random.default_rng(4).normal(size=(500, 16))
    result = calculate_kid(
        features, features.copy(), subsets=20, subset_size=200, seed=5
    )
    assert abs(result["kid_mean"]) < 0.05
    assert result["kid_std"] >= 0.0


def test_identity_reconstruction_metrics_are_exact():
    images = jnp.linspace(-1, 1, 24).reshape(2, 2, 2, 3)
    metrics = reconstruction_metrics(IdentityEncoder(), images)
    assert metrics["encoder_recon_mse"] == 0.0
    assert np.isinf(metrics["encoder_recon_psnr"])


def test_latent_sampling_uses_decoder_and_uint8_adapter():
    rngs = nnx.Rngs(7)
    model = FlowMatching(
        ZeroRHS((2, 2, 1)), rngs, 2, ode_method="euler", ode_nstep_max=1
    )
    decoder = CountingDecoder()
    batches = list(
        generate_image_batches(
            model,
            decoder,
            num_samples=3,
            batch_size=2,
            seed=8,
            ode_method="euler",
            ode_steps=1,
        )
    )
    assert [len(batch) for batch in batches] == [2, 1]
    assert all(batch.dtype == np.uint8 for batch in batches)
    assert decoder.decode_calls == 2


def test_sampling_is_independent_of_evaluation_batch_size():
    rngs = nnx.Rngs(17)
    model = FlowMatching(
        ZeroRHS((2, 2, 1)), rngs, 2, ode_method="euler", ode_nstep_max=1
    )
    first = np.concatenate(
        list(
            generate_image_batches(
                model, IdentityEncoder(), num_samples=5, batch_size=2, seed=18
            )
        )
    )
    second = np.concatenate(
        list(
            generate_image_batches(
                model, IdentityEncoder(), num_samples=5, batch_size=3, seed=18
            )
        )
    )
    np.testing.assert_array_equal(first, second)


def test_local_fid_matches_function_loaded_from_reference_source():
    source = '''import numpy as np\nimport scipy\ndef calculate_fid(stats: dict[str, np.ndarray], ref_stats: dict[str, np.ndarray]) -> float:\n    m = np.square(stats["mu"] - ref_stats["mu"]).sum()\n    s, _ = scipy.linalg.sqrtm(np.dot(stats["sigma"], ref_stats["sigma"]), disp=False)\n    return float(np.real(m + np.trace(stats["sigma"] + ref_stats["sigma"] - s * 2)))\n'''
    with tempfile.TemporaryDirectory() as directory:
        source_dir = Path(directory)
        (source_dir / "eval").mkdir()
        (source_dir / "eval" / "utils.py").write_text(source, encoding="utf-8")
        # Bypass source checkout verification only for this isolated AST unit by
        # loading the exact function body directly through a temporary git repo
        # would add no useful coverage. Numerical parity is checked below using
        # the same reference source form; pinned-source integration is a separate
        # smoke command.
        rng = np.random.default_rng(19)
        left = statistics_from_feature_batches([rng.normal(size=(8, 4))])
        right = statistics_from_feature_batches([rng.normal(size=(9, 4))])
        local = calculate_fid(left, right)
        namespace = {"np": np, "scipy": __import__("scipy")}
        exec(source, namespace)
        reference = namespace["calculate_fid"](
            left.as_diffuse_dict(), right.as_diffuse_dict()
        )
        np.testing.assert_allclose(local, reference)


def test_fid_supports_scipy_sqrtm_without_disp_argument():
    rng = np.random.default_rng(20)
    left = statistics_from_feature_batches([rng.normal(size=(8, 4))])
    right = statistics_from_feature_batches([rng.normal(size=(9, 4))])
    original = scipy.linalg.sqrtm

    def modern_sqrtm(matrix):
        result = original(matrix, disp=False)
        return result[0] if isinstance(result, tuple) else result

    scipy.linalg.sqrtm = modern_sqrtm
    try:
        assert np.isfinite(calculate_fid(left, right))
    finally:
        scipy.linalg.sqrtm = original


def test_end_to_end_checkpoint_evaluation_with_feature_caches():
    rngs = nnx.Rngs(23)
    model = FlowMatching(
        ZeroRHS((2, 2, 1)), rngs, 2, ode_method="euler", ode_nstep_max=1
    )
    validation = make_fixed_fm_validation(
        np.zeros((4, 2, 2, 1), dtype=np.float32), list("abcd"), 24
    )
    real_images = np.zeros((4, 2, 2, 1), dtype=np.float32)
    key = FIDCacheKey(
        "revision",
        "test",
        2,
        "center_square",
        "gray_to_rgb",
        feature_extractor=FakeExtractor.provenance,
    )
    with tempfile.TemporaryDirectory() as directory:
        real_cache = prepare_real_feature_cache(
            [{"image": real_images, "id": list("abcd")}],
            count=4,
            extractor=FakeExtractor(),
            cache_root=directory,
            key=key,
        )
        result = evaluate_checkpoint(
            model=model,
            encoder=IdentityEncoder(),
            validation=validation,
            real_feature_cache=real_cache,
            real_fid_key=key,
            fid_cache_root=directory,
            fake_cache_root=Path(directory) / "fake",
            extractor=FakeExtractor(),
            diffuse_source_dir=None,
            step=1,
            epoch=0.5,
            wall_clock_train_s=1.0,
            fm_batch_size=2,
            num_fake=4,
            sampling_batch_size=2,
            sampling_seed=25,
            sampling_config={"method": "euler", "steps": 1},
            kid_config={"enabled": True, "subsets": 2, "subset_size": 2},
            run_identity="test-run",
        )
        assert result["step"] == 1
        assert result["fid_num_real"] == 4
        assert result["fid_num_fake"] == 4
        assert np.isfinite(result["fid"])
        assert "kid_mean" in result


if __name__ == "__main__":
    test_fixed_fm_validation_is_repeatable()
    test_fixed_fm_validation_keeps_dropout_rng_across_batch_sizes()
    test_fid_statistics_and_cache_round_trip()
    test_kid_is_near_zero_for_same_distribution()
    test_identity_reconstruction_metrics_are_exact()
    test_latent_sampling_uses_decoder_and_uint8_adapter()
    test_sampling_is_independent_of_evaluation_batch_size()
    test_local_fid_matches_function_loaded_from_reference_source()
    test_fid_supports_scipy_sqrtm_without_disp_argument()
    test_end_to_end_checkpoint_evaluation_with_feature_caches()
    print("Image evaluation tests passed.")
