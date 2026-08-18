import jax
import jax.numpy as jnp
from flax import nnx

from neuripp.methods.anderson import get_anderson
from neuripp.methods.ngd import get_ngd
from neuripp.methods.optax_optimizer import get_optax
from neuripp.parametric_pushforward.flow_matching import FlowMatching, flow_matching_loss


class TinyMLP(nnx.Module):
    def __init__(self, rngs):
        self.dim = 2
        self.layers = nnx.Sequential(
            nnx.Linear(3, 4, rngs=rngs),
            nnx.swish,
            nnx.Linear(4, 2, rngs=rngs),
        )

    def __call__(self, time, x):
        return self.layers(jnp.concatenate((x, jnp.atleast_1d(time))))


def make_model(seed=0):
    rngs = nnx.Rngs(seed)
    return FlowMatching(TinyMLP(rngs), rngs, N_monte_carlo=4)


def test_interpolant_and_loss_shapes():
    model = make_model()
    rngs = nnx.Rngs(1)
    data = jnp.ones((4, 2))

    times, interpolants, latent = model.sample_interpolant(data, rngs, return_x0=True)

    assert times.shape == (4,)
    assert interpolants.shape == data.shape
    assert latent.shape == data.shape
    assert jnp.isfinite(flow_matching_loss(model, data, rngs))


def test_loss_is_vector_mean_squared_error():
    class ZeroRHS(nnx.Module):
        def __call__(self, time, x):
            return jnp.zeros_like(x)

    class FixedModel(nnx.Module):
        def __init__(self):
            self.rhs = ZeroRHS()

        def sample_interpolant(self, data_batch, rngs, return_x0=False):
            times = jnp.zeros(data_batch.shape[0])
            interpolants = jnp.zeros_like(data_batch)
            latent = jnp.zeros_like(data_batch)
            return (times, interpolants, latent) if return_x0 else (times, interpolants)

    loss = flow_matching_loss(
        FixedModel(),
        jnp.asarray([[1.0, -1.0]]),
        nnx.Rngs(0),
    )
    assert jnp.isclose(loss, 2.0)


def test_all_optimizers_complete_a_step():
    data = jnp.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    optimizers = {
        "ngd": lambda: (
            *get_ngd(flow_matching_loss),
            (1e-3,),
            {"linear_solver_maxiter": 2},
        ),
        "anderson": lambda: (
            *get_anderson(flow_matching_loss, history_length=2),
            (1e-3, 1.0, 1e-3),
            {"linear_solver_maxiter": 2},
        ),
        "adam": lambda: (*get_optax(flow_matching_loss, method="adam"), (1e-3,), {}),
        "sgd": lambda: (*get_optax(flow_matching_loss, method="sgd"), (1e-3,), {}),
    }

    for index, factory in enumerate(optimizers.values()):
        model = make_model(index + 10)
        rngs = nnx.Rngs(index + 20)
        init_fn, step_fn, optimizer_args, optimizer_kwargs = factory()
        state = init_fn(model, optimizer_args, optimizer_kwargs, data, rngs)
        state, values = step_fn(state, data, rngs)

        assert len(values) in (2, 3)
        assert all(bool(jnp.isfinite(value)) for value in values)
        leaves = jax.tree.leaves(nnx.state(state[0], nnx.Param))
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


if __name__ == "__main__":
    test_interpolant_and_loss_shapes()
    test_loss_is_vector_mean_squared_error()
    test_all_optimizers_complete_a_step()
    print("Flow-matching tests passed.")
