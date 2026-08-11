"""Defines a model with the geometry induced by flow matching"""

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.utility.utility import tree_dot_product

ZERO_TOL = 1e-20


def _dropout_keys(rhs, rngs: nnx.Rngs | None, count: int):
    if rngs is None or not getattr(rhs, "uses_explicit_dropout_rng", False):
        return None
    return jax.random.split(rngs(), count)


def _batched_rhs(rhs, times, states, dropout_keys=None):
    """Vectorize an RHS with explicit, differentiation-safe dropout keys."""

    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def deterministic_predict(current_rhs, current_time, current_state):
        return current_rhs(current_time, current_state)

    @nnx.vmap(in_axes=(None, 0, 0, 0), out_axes=0)
    def stochastic_predict(current_rhs, current_time, current_state, dropout_key):
        return current_rhs(current_time, current_state, dropout_key)

    if dropout_keys is None:
        return deterministic_predict(rhs, times, states)
    return stochastic_predict(rhs, times, states, dropout_keys)


class FlowMatching(ParametricPushforward):

    def _sample_latent(self, N_samples: int, rngs: nnx.Rngs):
        """Sample Gaussian states while preserving the RHS state shape.

        ``ParametricPushforward`` historically flattens latent samples because
        its density-estimation path augments vectors along their last axis.
        Flow Matching does not need that augmentation during training or
        generation, and image vector fields need their native ``(H, W, C)``
        state shape.
        """

        return rngs.normal((N_samples, *self.dim))

    def sample_interpolant(
        self,
        data_batch: jnp.ndarray,
        rngs: nnx.Rngs | None,
        return_x0: bool = False,
        *,
        times: jnp.ndarray | None = None,
        noise: jnp.ndarray | None = None,
    ):
        """Construct noise-to-data linear interpolants.

        Supplying ``times`` and ``noise`` makes the objective independent of
        mutable RNG state.  Evaluation uses this path to compare checkpoints on
        exactly the same held-out Flow Matching problem.
        """

        data_batch = jnp.asarray(data_batch)
        if data_batch.ndim < 2:
            raise ValueError(
                "Flow Matching data must have a leading batch axis and at least "
                "one state axis"
            )
        n_samples = data_batch.shape[0]
        x1 = data_batch
        if noise is None:
            if rngs is None:
                raise ValueError("rngs is required when noise is not provided")
            x0 = self._sample_latent(n_samples, rngs)
        else:
            x0 = jnp.asarray(noise, dtype=data_batch.dtype)
            if x0.shape != data_batch.shape:
                raise ValueError(
                    f"noise shape {x0.shape} must match data shape {data_batch.shape}"
                )

        if times is None:
            if rngs is None:
                raise ValueError("rngs is required when times are not provided")
            ts = rngs.uniform((n_samples,), dtype=data_batch.dtype)
        else:
            ts = jnp.asarray(times, dtype=data_batch.dtype)
            if ts.shape != (n_samples,):
                raise ValueError(
                    f"times shape {ts.shape} must be ({n_samples},)"
                )

        time_shape = (n_samples,) + (1,) * (data_batch.ndim - 1)
        ts_broadcast = ts.reshape(time_shape)
        xts = x0 * (1.0 - ts_broadcast) + x1 * ts_broadcast

        if return_x0:
            return ts, xts, x0

        return ts, xts

    def scalar_product(
        self,
        tangent1: PyTree,
        tangent2: PyTree,
        rngs: nnx.Rngs,
        data_batch: jnp.ndarray | None = None,
        interpolant=None,
    ):
        """Computes the scalar product of tangent vectors in the pullback Wasserstein metric

        .. note::

            Tangents should be trainable parameters, e.g. output of ``nnx.grad``

        """
        if interpolant is None:
            if data_batch is None:
                raise ValueError("data_batch is required for the flow-matching metric")
            interpolant = self.sample_interpolant(data_batch, rngs)

        ts, xts = interpolant
        dropout_keys = _dropout_keys(self.rhs, rngs, ts.shape[0])

        gd, params, rest = nnx.split(self, nnx.Param, ...)

        def _T(_par):
            _model = nnx.merge(gd, _par, rest)
            return _batched_rhs(_model.rhs, ts, xts, dropout_keys)

        _, dT_dtheta = jax.linearize(_T, params)

        dT_dtang1 = dT_dtheta(tangent1)
        dT_dtang2 = dT_dtheta(tangent2)

        return tree_dot_product(dT_dtang1, dT_dtang2) / xts.shape[0]

    def get_matvec_fn(
        self,
        rngs: nnx.Rngs,
        data_batch: jnp.ndarray | None = None,
        interpolant=None,
    ):
        """For fixed set of parameters, generates latent samples and gives a function that computes :maht:`G(\\theta)\\mathrm{d}\\theta`"""
        if interpolant is None:
            if data_batch is None:
                raise ValueError("data_batch is required for the flow-matching metric")
            interpolant = self.sample_interpolant(data_batch, rngs)

        ts, xts = interpolant
        dropout_keys = _dropout_keys(self.rhs, rngs, ts.shape[0])
        gd, params, rest = nnx.split(self, nnx.Param, ...)

        def _T(_par):
            _model = nnx.merge(gd, _par, rest)
            return _batched_rhs(
                _model.rhs, ts, xts, dropout_keys
            ) / jnp.sqrt(ts.shape[0])

        _, dT_dtheta = jax.linearize(_T, params)
        dT_transpose_dtheta = jax.linear_transpose(dT_dtheta, params)

        def _matvec_fn(tang: dict):
            (matvec,) = dT_transpose_dtheta(dT_dtheta(tang))
            return matvec

        return _matvec_fn


def flow_matching_loss(
    model: FlowMatching,
    data_batch: jnp.ndarray,
    rngs: nnx.Rngs | None,
    times: jnp.ndarray | None = None,
    noise: jnp.ndarray | None = None,
    dropout_keys: jnp.ndarray | None = None,
):
    """Return the mean per-example squared Flow Matching error.

    Every non-batch state axis is reduced, preserving the historical vector
    objective while extending it to images and spatial latent tensors.
    """

    interpolant_kwargs = {}
    if times is not None:
        interpolant_kwargs["times"] = times
    if noise is not None:
        interpolant_kwargs["noise"] = noise
    ts, xts, x0 = model.sample_interpolant(
        data_batch, rngs, return_x0=True, **interpolant_kwargs
    )
    x1 = data_batch
    if dropout_keys is None:
        dropout_keys = _dropout_keys(model.rhs, rngs, ts.shape[0])
    vs = _batched_rhs(model.rhs, ts, xts, dropout_keys)
    v_empirical = x1 - x0
    state_axes = tuple(range(1, vs.ndim))
    return jnp.mean(jnp.sum((vs - v_empirical) ** 2, axis=state_axes))
