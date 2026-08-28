import jax.numpy as jnp
import numpy as np
from flax import nnx

from neuripp.parametric_pushforward.flow_matching import (
    FlowMatching,
    flow_matching_loss,
)


class ZeroImageRHS(nnx.Module):
    def __init__(self, shape):
        self.dim = shape

    def __call__(self, time, state, *args, rngs=None):
        del time
        del args
        del rngs
        return jnp.zeros_like(state)


def make_image_model(shape=(4, 5, 3)):
    rngs = nnx.Rngs(0)
    return FlowMatching(
        ZeroImageRHS(shape),
        rngs,
        N_monte_carlo=2,
        ode_method="euler",
        ode_nstep_max=2,
    )


def test_spatial_interpolant_preserves_nhwc_shape():
    model = make_image_model()
    data = jnp.ones((2, 4, 5, 3))
    times = jnp.asarray([0.0, 1.0])
    noise = jnp.zeros_like(data)

    returned_times, interpolants, returned_noise = model.sample_interpolant(
        data,
        None,
        return_x0=True,
        times=times,
        noise=noise,
    )

    assert returned_times.shape == (2,)
    assert interpolants.shape == data.shape
    np.testing.assert_allclose(interpolants[0], noise[0])
    np.testing.assert_allclose(interpolants[1], data[1])
    np.testing.assert_allclose(returned_noise, noise)


def test_spatial_loss_reduces_all_state_axes():
    model = make_image_model(shape=(2, 3, 1))
    data = jnp.ones((2, 2, 3, 1))
    noise = jnp.zeros_like(data)
    times = jnp.asarray([0.25, 0.75])

    loss = flow_matching_loss(
        model,
        data,
        None,
        times=times,
        noise=noise,
    )

    # The zero vector field differs from the all-ones target in six entries.
    np.testing.assert_allclose(loss, 6.0)


def test_fixed_interpolant_is_rng_independent():
    model = make_image_model(shape=(2, 2, 1))
    data = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2, 1)
    noise = -jnp.ones_like(data)
    times = jnp.asarray([0.2, 0.8])

    first = flow_matching_loss(model, data, None, times=times, noise=noise)
    second = flow_matching_loss(model, data, None, times=times, noise=noise)
    np.testing.assert_array_equal(first, second)


def test_spatial_sampling_preserves_state_shape():
    model = make_image_model(shape=(2, 3, 1))
    samples = model.sample(3, nnx.Rngs(4))
    assert samples.shape == (3, 2, 3, 1)


if __name__ == "__main__":
    test_spatial_interpolant_preserves_nhwc_shape()
    test_spatial_loss_reduces_all_state_axes()
    test_fixed_interpolant_is_rng_independent()
    test_spatial_sampling_preserves_state_shape()
    print("Image Flow Matching tests passed.")
