"""Defines a model with the geometry induced by flow matching"""

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from flax import nnx

from neuripp._ode._ode import solve_ode_batched
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.utility.utility import tree_dot_product

from typing import Literal

ZERO_TOL = 1e-20


class FlowMatching(ParametricPushforward):

    def sample_interpolant(
        self, data_batch: jnp.ndarray, rngs: nnx.Rngs, return_x0=False
    ):
        n_samples = data_batch.shape[0]
        x1 = data_batch
        x0 = self._sample_latent(n_samples, rngs)
        ts = rngs.uniform()
        xts = x0 * (1.0 - ts)[:, ...] + x1 * ts[:, ...]

        if return_x0:
            return ts, xts, x0

        return ts, xts

    def scalar_product(
        self,
        data_batch: jnp.array,
        tangent1: PyTree,
        tangent2: PyTree,
        rngs: nnx.Rngs,
        interpolant=None,
    ):
        """Computes the scalar product of tangent vectors in the pullback Wasserstein metric

        .. note::

            Tangents should be trainable parameters, e.g. output of ``nnx.grad``

        """
        if interpolant is None:
            interpolant = self.sample_interpolant(data_batch, rngs)

        ts, xts = interpolant

        gd, params, rest = nnx.split(self, nnx.Param, ...)

        def _T(_par):
            _model = nnx.merge(gd, _par, rest)
            return _model.rhs(ts, xts)

        _, dT_dtheta = jax.linearize(_T, params)

        dT_dtang1 = dT_dtheta(tangent1)
        dT_dtang2 = dT_dtheta(tangent2)

        return tree_dot_product(dT_dtang1, dT_dtang2) / xts.shape[0]

    def get_matvec_fn(
        self,
        rngs: nnx.Rngs,
        interpolant=None,
    ):
        """For fixed set of parameters, generates latent samples and gives a function that computes :maht:`G(\\theta)\\mathrm{d}\\theta`"""
        if interpolant is None:
            interpolant = self.sample_interpolant(data_batch, rngs)

        gd, params, rest = nnx.split(self, nnx.Param, ...)

        def _T(_par):
            _model = nnx.merge(gd, _par, rest)
            return _model.rhs(*interpolant) / jnp.sqrt(interpolant[0].shape[0])

        _, dT_dtheta = jax.linearize(_T, params)
        dT_transpose_dtheta = jax.linear_transpose(dT_dtheta, params)

        def _matvec_fn(tang: dict):
            (matvec,) = dT_transpose_dtheta(dT_dtheta(tang))
            return matvec

        return _matvec_fn


def flow_matching_loss(model: FlowMatching, data_batch: jnp.ndarray, rngs: nnx.Rngs):
    ts, xts, x0 = model.sample_interpolant(data_batch, rngs, return_x0=True)
    x1 = data_batch
    vs = nnx.vmap(model.rhs)(ts, xts)
    v_empirical = x1 - x0

    return jnp.mean(((vs - v_empirical).sum(axis=-1)) ** 2)
