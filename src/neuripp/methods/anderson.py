from typing import Callable, Union, Generator, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial

from ..problems.problem import Problem
from ..parametric_pushforward.parametric_pushforward import ParametricPushforward



# Naming tentative; Riemannian -> Wasserstein? Ottonian? Steinian?
class RAMSolver:
    def __init__(
        self,
        problem: Problem,
        relaxation: Union[jnp.float64, Generator] = 0.95,
        l2_regularization: Union[jnp.float64, Generator] = 0.0,
        history_len: int = 2,
    ):
        self._problem : Problem = problem
        # self._k = 0
        self._m = history_len

        # QUESTION: how does it actually affect the jit / performance (in comparison to just constant)?
        self._relaxation = _as_generator(relaxation)
        self._l2_reg = _as_generator(l2_regularization)

        self._x_cur, self._x_prev, self._r_prev, self._delta_rs, self._delta_xs = (
            None,
        ) * 5

    # TODO: ? pass any arguments such as seed to .suggest_initial
    def _initialize_iteration(self, *args, **kwargs):
        x0: ParametricPushforward = self._problem.suggest_initial(*args, **kwargs)
        r0 = self._problem.residual(x0)
        x1 = x0.riemannian_exp(r0)


        self._x_prev = x0.copy()  # x_k-1
        self._x_cur = x1.copy()  # x_k
        self._r_prev = r0.copy()

        # self._delta_rs = None
        # TODO: unlike arrays, I don't know if there is a way to stack
        # m zeros-like kind of trees and then then conveniently index them
        self._delta_rs = jnp.zeros((N, d + 1, self._m))
        # self._delta_xs = r0.copy()[:,  + 1:, jnp.newaxis]
        self._delta_xs = jnp.zeros((N, d + 1, self._m))
        self._delta_xs = self._delta_xs.at[:, :, 0].set(r0)
        # self._k = 1
        metric_vals = (residual_rkhs, dx_l2)

        if self._metrics is not None:
            metric_vals += tuple(m(x1) for m in self._metrics)

        return self, metric_vals

    # TODO jit ?
    def _step(
        self,
    ):
        rk = self._operator(self._x_cur)
        # Compute Gram matrix Gk of the current basis
        Gk = pairwiseScalarProductOfBasisVectors(self._x_cur, self._x_cur, self._kernel)
        # And auxilary matrix Tk, needed to compute projections
        Tk = pairwiseScalarProductOfBasisVectors(
            self._x_cur, self._x_prev, self._kernel
        )

        # Transport Delta X and Delta r vectors to the tangent space of the current iter
        # TODO: vector transport r_k-1, R_k, X_k in one vectorized solve!
        self._delta_xs = vectorTransport(
            self._x_cur, Gk, self._x_prev, self._delta_xs[:, :, : self._m], Tk
        )
        delta_r_cur = (
            rk
            - vectorTransport(self._x_cur, Gk, self._x_prev, self._r_prev, Tk)[:, :, 0]
        )
        self._delta_rs = self._delta_rs.at[:, :, 1:].set(
            vectorTransport(
                self._x_cur, Gk, self._x_prev, self._delta_rs[:, :, : self._m - 1], Tk
            )
        )
        self._delta_rs = self._delta_rs.at[:, :, 0].set(delta_r_cur)

        R = self._delta_rs
        X = self._delta_xs

        # TODO l_infty CONSTRAINED minimization?
        #       or adaptive regularization
        lam = next(self._l2_reg)
        W_quad = jnp.einsum("ijm,ijkl,kln->mn", R, Gk, R) + lam * jnp.eye(self._m)
        rhs = jnp.einsum("ij,ijkl,klm->m", rk, Gk, R)
        # for small matrix < 15 x 15 direct solve should be OK
        Gamma = jnp.linalg.solve(W_quad, rhs)
        Gamma = jnp.atleast_1d(Gamma)

        rk_bar = rk - R @ Gamma
        delta_x_cur = -X @ Gamma + next(self._relaxation) * rk_bar

        self._delta_xs = jnp.roll(self._delta_xs, 1, axis=-1)
        self._delta_xs = self._delta_xs.at[:, :, 0].set(delta_x_cur)

        self._x_prev = self._x_cur.copy()
        v = evalTangent(
            self._x_cur, delta_x_cur, self._x_cur, self._kernel
        )  # <<Exponential>>

        self._x_cur += v
        self._r_prev = rk

        residual_rkhs = norm_rkhs(rk, Gk)

        dx_l2 = norm_l2(v)
        metric_vals = jnp.array([residual_rkhs, dx_l2])

        if self._metrics is not None:
            # metric_vals += tuple(m(self._x_cur) for m in self._metrics)
            additional_metrics = jax.lax.map(lambda m: m(self._x_cur), self._metrics)
            metrics_vals = jnp.concat((metric_vals, additional_metrics))

        return self, metric_vals

    @partial(jax.jit, static_argnames="max_iter")
    def iterate(
        self,
        x0: jnp.ndarray,
        max_iter: int = 10,
    ):
        solver, metric_vals_orig = self._initialize_iteration(x0)
        metric_vals = [metric_vals_orig]
        metric_vals = jnp.array(metric_vals)

        def body_fn(solver, *args):
            solver, metric_vals = solver._step()

            return solver, metric_vals

        solver, metric_vals_run = jax.lax.scan(body_fn, solver, length=max_iter)

        metric_vals_run = jnp.array(metric_vals_run)
        metric_vals = jnp.concatenate((metric_vals, metric_vals_run), axis=0).T

        return solver, metric_vals

    def _tree_flatten(self):
        children = (
            self._x_cur,
            self._x_prev,
            self._r_prev,
            self._delta_rs,
            self._delta_xs,
        )
        aux_data = {
            "history_len": self._m,
            "operator": self._operator,
            "kernel": self._kernel,
            "relaxation": self._relaxation,
            "l2_regularization": self._l2_reg,
            "metrics": self._metrics,
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        sol = cls(**aux_data)
        sol._x_cur, sol._x_prev, sol._r_prev, sol._delta_rs, sol._delta_xs = children

        return sol


jax.tree_util.register_pytree_node(
    RAMSolver, RAMSolver._tree_flatten, RAMSolver._tree_unflatten
)
