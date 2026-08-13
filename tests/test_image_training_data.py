from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from image_benchmarks.training.data import RestartableLatentStream


def _stream(mean, std, *, seed=1, sampling_seed=2, sample_posterior=True):
    return RestartableLatentStream(
        mean,
        std,
        batch_size=2,
        seed=seed,
        sampling_seed=sampling_seed,
        sample_posterior=sample_posterior,
        shuffle=True,
        drop_last=True,
    )


def test_latent_shuffle_and_posterior_rng_streams_are_separate():
    mean = np.arange(8, dtype=np.float32).reshape(4, 2)
    deterministic_a = _stream(mean, np.zeros_like(mean), sampling_seed=2).next_batch()
    deterministic_b = _stream(mean, np.zeros_like(mean), sampling_seed=99).next_batch()
    np.testing.assert_array_equal(deterministic_a, deterministic_b)

    stochastic_a = _stream(mean, np.ones_like(mean), sampling_seed=2).next_batch()
    stochastic_b = _stream(mean, np.ones_like(mean), sampling_seed=99).next_batch()
    assert not np.array_equal(stochastic_a, stochastic_b)
    posterior_mean = _stream(
        mean, np.ones_like(mean), sampling_seed=99, sample_posterior=False
    ).next_batch()
    np.testing.assert_array_equal(posterior_mean, deterministic_a)


def test_latent_stream_resume_restores_exact_next_batch():
    mean = np.arange(16, dtype=np.float32).reshape(8, 2)
    std = np.ones_like(mean) * 0.1
    original = _stream(mean, std)
    original.next_batch()
    state = original.state_dict()
    expected = original.next_batch()

    resumed = _stream(mean, std)
    resumed.load_state_dict(state)
    np.testing.assert_array_equal(resumed.next_batch(), expected)


if __name__ == "__main__":
    test_latent_shuffle_and_posterior_rng_streams_are_separate()
    test_latent_stream_resume_restores_exact_next_batch()
    print("Image training data tests passed.")
