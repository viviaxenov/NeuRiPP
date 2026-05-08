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

from flax import nnx

from neuripp._ode._ode import solve_ode_batched

from typing import Literal, Callable

ZERO_TOL = 1e-20


class ParametricPushforward(nnx.Module):
    def __init__(
        self,
        rhs: nnx.Module,
        N_monte_carlo: int,
        seed: int,
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
        self._rngs = nnx.Rngs(seed)
        assert hasattr(
            rhs, "dim"
        ), "RHS must have a `dim` property for the dimension of `x`"

        # TODO: if rhs doesn't accept *args, do a wrapper?
        self.rhs = rhs

        self._N_mc = N_monte_carlo
        self.ode_nstep_max = ode_nstep_max
        self.ode_method = ode_method
        self.ode_kwargs = ode_kwargs if ode_kwargs is not None else dict()
        self.div_method = divergence_method

    def __call__(self, z: jnp.ndarray):
        return solve_ode_batched(
            self.rhs,
            z,
            N_steps_max=self.ode_nstep_max,
            method=self.ode_method,
            **self.ode_kwargs
        )

    def rhs_reverse_time(self, t, x, *args):
        return -self.rhs(1.0 - t, x, *args)

    def rhs_div(self, t, x, *args):
        dxdt, jvp_fn = jax.linearize(lambda _x: self.rhs(t, _x), x[..., :-1])
        vects = jnp.eye(dxdt.shape[-1])
        jac_tr = jnp.sum(jax.vmap(lambda eps: eps.T @ jvp_fn(eps))(vects))

        return jnp.concat((dxdt, -jnp.atleast_1d(jac_tr)), axis=-1)

    def rhs_div_hutchinson(self, t, x, eps):
        dxdt, jvp_eps = jax.jvp(lambda _x: self.rhs(t, _x), (x[..., :-1],), (eps,))
        jac_tr = eps.T @ jvp_eps
        # print(
        #     dxdt.shape,
        #     eps.shape,
        #     jvp_eps.shape,
        #     jac_tr.shape,
        #     flush=True
        # )
        return jnp.concat((dxdt, -jnp.atleast_1d(jac_tr)), axis=-1)

    def pushforward(*args, **kwargs):
        return self(*args, **kwargs)

    def pullback(self, x: jnp.ndarray):
        """Compute inverse of the parametric mapping"""
        return solve_ode_batched(
            self.rhs_reverse_time,
            x,
            N_steps_max=self.ode_nstep_max,
            method=self.ode_method,
            **self.ode_kwargs
        )

    def _sample_latent(self, N_samples: int):
        return self._rngs.normal((N_samples, self.rhs.dim))

    def _latent_log_density(self, z: jnp.ndarray):
        """Returns the log density of the latent distribution"""
        return -0.5 * ((z.reshape(z.shape[0], -1)) ** 2).sum(axis=-1)

    def sample(self, N_samples: int, with_log_density=False):
        """Returns a sample `x` of shape `(N_samples, dim)` from the current distribution :math:`\\rho_\\theta`"""
        x0 = self._sample_latent(N_samples)
        rhs_of_system = self.rhs
        aux_args = None
        if with_log_density:
            # Augment system and initial condition
            logp_latent = self._latent_log_density(x0)
            if self.div_method == "exact":
                rhs_of_system = self.rhs_div
            else:
                eps = self._rngs.rademacher(x0.shape, dtype=x0.dtype)
                rhs_of_system = self.rhs_div_hutchinson
                aux_args = eps
            x0 = jnp.concat((x0, logp_latent.reshape(-1, 1)), axis=-1)

        res = solve_ode_batched(
            rhs_of_system,
            x0,
            aux_args,
            N_steps_max=self.ode_nstep_max,
            method=self.ode_method,
            **self.ode_kwargs
        )

        if with_log_density:
            x, logp_of_x = res[:, :-1], res[:, -1]
            return x, logp_of_x

        return res

    # TODO: type for parameters and tangent?
    # m.b. use PyTree as parameters and always merge?
    # need to understand when untrainable params are updated e.g. batch norm
    def scalar_product(self, tangent1, tangent2, N_monte_carlo=None, param=None):
        """Computes the scalar product of tangent vectors in the pullback Wasserstein metric

        .. note::

            Tangents should be trainable parameters, e.g. output of ``nnx.grad``

        """
        pass

    def riemann_tensor_matvec(self, tangent, param=None):
        pass

    def norm(self, tangent, N_monte_carlo=None, param=None):
        norm_sq = self.scalar_product(tangent, N_monte_carlo, param)
        return jnp.sqrt(jnp.maximum(norm_sq, ZERO_TOL))

    def riemannian_exp(self, tangent, param=None):
        if param is None:
            param = nnx.split(
                self,
            )

        new_param = jax.tree.map(lambda _x, _v: _x + _v, param, tangent)
        nnx.update(self, new_param)

    def vector_transport(self, tangent, param_new, param_cur=None):
        return tangent
