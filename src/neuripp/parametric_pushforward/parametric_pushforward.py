"""Defines a Parametric Pushforward model

Parametric Pushforward is a model for a multidimensional probability distribution.
A fixed latent density (as for now, standard Gaussian) is pushed forward by a mapping, defined by trainable parameters.
As for now, the mapping is a Neural ODE with variable architectures in the right-hand side.
Sampling, computing functionals such as :math:`KL` divergence, is supported.

The model can be used in generative modelling or Bayesian inverse problems.

It can be trained with Riemannian methods with respect to the pullback of the Wasserstein metric:

.. math::

    \\langle \\dot \\theta_1, \\theta_2 \\rangle_{\\theta}
        = \int \left\langle \\frac{\partial T_\\theta(x)}{\partial \\theta} \cdot \\theta_1, \\frac{\partial T_\\theta(x)}{\partial \\theta} \cdot \\theta_2\\right\\rangle  \\mathrm{d}\\rho_\\theta(x)


where :math:`T_\\theta` is the mapping, given by the set of parameters :math:`\\theta`, :math:`\\rho_\\theta = (T_\\theta)_\\sharp\\rho_\\text{ref}` is the pusforward densitites, :math:`\\theta_{1,2}` are the tangent vectors in the parameter space, and :math:`\\frac{\partial T_\\theta(x)}{\partial \\theta} \cdot \\theta_{1,2}` is a directional derivative in the direction of :math:`\\theta_{1,2}`
"""

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from flax import nnx

from neuripp._ode._ode import solve_ode_batched
from neuripp.utility.utility import tree_dot_product

from typing import Literal

ZERO_TOL = 1e-20


class ParametricPushforward(nnx.Module):
    def __init__(
        self,
        rhs: nnx.Module,
        rngs: nnx.Rngs,
        N_monte_carlo: int,
        ode_nstep_max: int = 100,
        ode_method: str = "rk45",
        ode_kwargs: dict = None,
        divergence_method: Literal["exact", "hutchinson"] = "hutchinson",
    ):
        """
        .. note::

            Assumes that rhs(t, x) accepts x's of shape (n_batch, dim). For images/other structured data, reshaping should be done by RHS.
            Maybe need to provide a wrapper.
        """
        assert hasattr(
            rhs, "dim"
        ), "RHS must have a `dim` property for the dimension of `x`"

        assert isinstance(rhs.dim, (int, tuple)), f"rhs.dim must be int or Tuple[int], but got {rhs.dim}"

        # TODO: if rhs doesn't accept *args, do a wrapper?
        self.rhs = rhs
        self.dim = rhs.dim
        if isinstance(rhs.dim, int):
            self.dim = (rhs.dim,)

        self._N_mc = N_monte_carlo
        self.ode_nstep_max = ode_nstep_max
        self.ode_method = ode_method
        self.ode_kwargs = ode_kwargs if ode_kwargs is not None else dict()
        self.div_method = divergence_method

    def __call__(self, z: jnp.ndarray, rngs: nnx.Rngs=None, with_log_density=False):
        rhs_of_system = self.rhs
        aux_args = None
        if with_log_density:
            # Augment system and initial condition
            logp_latent = self._latent_log_density(z)
            if self.div_method == "exact":
                rhs_of_system = self.rhs_div
            else:
                eps = rngs.rademacher(z.shape, dtype=z.dtype)
                rhs_of_system = self.rhs_div_hutchinson
                aux_args = eps
            z = jnp.concat((z, logp_latent.reshape(-1, 1)), axis=-1)

        res = solve_ode_batched(
            rhs_of_system,
            z,
            aux_args,
            N_steps_max=self.ode_nstep_max,
            method=self.ode_method,
            **self.ode_kwargs
        )

        if with_log_density:
            x, logp_of_x = res[:, :-1], res[:, -1]
            return x, logp_of_x

        return res

    def rhs_reverse_time(self, t, x, *args):
        return -self.rhs(1.0 - t, x, *args)

    def rhs_div(self, t, x, *args):
        dxdt, jvp_fn = jax.linearize(lambda _x: self.rhs(t, _x), x[..., :-1])
        vects = jnp.eye(dxdt.shape[-1])
        jac_tr = jnp.sum(jax.vmap(lambda eps: eps.T @ jvp_fn(eps))(vects))

        return jnp.concat((dxdt, -jnp.atleast_1d(jac_tr)), axis=-1)

    def rhs_div_inv_time(self, t, x, *args):
        dxdt, jvp_fn = jax.linearize(lambda _x: -self.rhs(1.0 - t, _x), x[..., :-1])
        vects = jnp.eye(dxdt.shape[-1])
        jac_tr = jnp.sum(jax.vmap(lambda eps: eps.T @ jvp_fn(eps))(vects))

        return jnp.concat((dxdt, -jnp.atleast_1d(jac_tr)), axis=-1)

    def rhs_div_hutchinson(self, t, x, eps):
        dxdt, jvp_eps = jax.jvp(lambda _x: self.rhs(t, _x), (x[..., :-1],), (eps,))
        jac_tr = eps.T @ jvp_eps
        return jnp.concat((dxdt, -jnp.atleast_1d(jac_tr)), axis=-1)

    def rhs_div_hutchinson_inv_time(self, t, x, eps):
        dxdt, jvp_eps = jax.jvp(
            lambda _x: -self.rhs(1.0 - t, _x), (x[..., :-1],), (eps,)
        )
        jac_tr = eps.T @ jvp_eps
        return jnp.concat((dxdt, -jnp.atleast_1d(jac_tr)), axis=-1)

    def pushforward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def pullback(self, x: jnp.ndarray, rngs: nnx.Rngs=None, with_log_density=False):
        """Compute inverse of the parametric mapping"""
        rhs_of_system = self.rhs
        aux_args = None
        if with_log_density:
            # Augment system and initial condition
            logp_latent = jnp.zeros(x.shape[0])
            if self.div_method == "exact":
                rhs_of_system = self.rhs_div_inv_time
            else:
                eps = rngs.rademacher(x.shape, dtype=x.dtype)
                rhs_of_system = self.rhs_div_hutchinson_inv_time
                aux_args = eps
            x = jnp.concat((x, logp_latent.reshape(-1, 1)), axis=-1)

        res = solve_ode_batched(
            rhs_of_system,
            x,
            aux_args,
            N_steps_max=self.ode_nstep_max,
            method=self.ode_method,
            **self.ode_kwargs
        )

        if with_log_density:
            z, d_logp = res[:, :-1], res[:, -1]
            logp_of_x = self._latent_log_density(z) - d_logp
            return z, logp_of_x

        return res

    def _sample_latent(self, N_samples: int, rngs: nnx.Rngs):
        return rngs.normal((N_samples, *self.dim)).reshape(N_samples, -1)

    def _latent_log_density(self, z: jnp.ndarray):
        """Returns the log density of the latent distribution"""
        return -0.5 * ((z.reshape(z.shape[0], -1)) ** 2).sum(axis=-1) # - 0.5*z.shape[-1]*jnp.log(2.*jnp.pi)

    def sample(self, N_samples: int, rngs: nnx.Rngs, with_log_density=False):
        """Returns a sample `x` of shape `(N_samples, dim)` from the current distribution :math:`\\rho_\\theta`"""
        z = self._sample_latent(N_samples, rngs)
        return self(z, rngs, with_log_density=with_log_density)

    def scalar_product(
        self,
        tangent1: PyTree,
        tangent2: PyTree,
        rngs: nnx.Rngs
    ):
        """Computes the scalar product of tangent vectors in the pullback Wasserstein metric

        .. note::

            Tangents should be trainable parameters, e.g. output of ``nnx.grad``

        """
        z = self._sample_latent(self._N_mc, rngs)
        gd, params, rest = nnx.split(self, nnx.Param, ...)

        def _T(_par):
            _model = nnx.merge(gd, _par, rest)
            return _model(z)

        _, dT_dtheta = jax.linearize(_T, params)

        dT_dtang1 = dT_dtheta(tangent1)
        dT_dtang2 = dT_dtheta(tangent2)

        return tree_dot_product(dT_dtang1, dT_dtang2) / z.shape[0]

    def get_matvec_fn(
        self,
        rngs: nnx.Rngs,
    ):
        """For fixed set of parameters, generates latent samples and gives a function that computes :maht:`G(\\theta)\\mathrm{d}\\theta`"""
        z = self._sample_latent(self._N_mc, rngs)
        gd, params, rest = nnx.split(self, nnx.Param, ...)

        def _T(_par):
            _model = nnx.merge(gd, _par, rest)
            return _model(z) / jnp.sqrt(z.shape[0])

        _, dT_dtheta = jax.linearize(_T, params)
        dT_transpose_dtheta = jax.linear_transpose(dT_dtheta, params)

        def _matvec_fn(tang: dict):
            (matvec,) = dT_transpose_dtheta(dT_dtheta(tang))
            return matvec

        return _matvec_fn

    def norm(self, tangent: PyTree, rngs):
        norm_sq = self.scalar_product(tangent, tangent,  rngs)
        return jnp.sqrt(jnp.maximum(norm_sq, ZERO_TOL))
