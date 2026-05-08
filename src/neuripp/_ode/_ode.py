import abc
from typing import Callable, Tuple, Literal, Union
import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


class ODEStep(abc.ABC):
    """Base class for ODE solvers."""

    # ! TODO: all of them MUST support h_min and guarantee that __call__ advances the integration at least by h_min!

    def __init__(self, rhs, *args, **kwargs):
        self._rhs = rhs

    @abc.abstractmethod
    def __call__(
        self,
        t: jnp.float32,
        x: jnp.ndarray,
        h: jnp.float32,
        **kwargs,
    ) -> Tuple[float, jnp.ndarray, float]:
        """Perform one integration step."""
        pass

    def suggest_h0(
        self,
        h_min: float,
    ):
        return h_min

    @abc.abstractmethod
    def tree_flatten(self):
        pass

    @abc.abstractclassmethod
    def tree_unflatten(cls, aux_data, children):
        pass


@jax.tree_util.register_pytree_node_class
class RK45Step(ODEStep):
    # Butcher's tableau data
    A = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ]
    )
    B = jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    C = jnp.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
    E = jnp.array(
        [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40]
    )
    n_stages = A.shape[0]
    butcher_tableau = (A, B, C, E)

    def __init__(
        self,
        rhs: Callable,
        rtol: jnp.float32 = 1e-4,
        atol: jnp.float32 = 1e-16,
        h_min: jnp.float32 = 1e-10,
        h_max: jnp.float32 = 0.2,
        N_iter_to_accept: int = 2,
    ):
        self._rhs = rhs
        self._rtol = rtol
        self._atol = atol
        self._h_min = h_min
        self._h_max = h_max
        self._N_iter_to_accept = N_iter_to_accept

    def __call__(
        self,
        t: jnp.float32,
        x: jnp.ndarray,
        h_cur: jnp.float32,
        *args,
    ):
        # TODO: put in main loop, check so that is in the right place
        h_current = jnp.minimum(h_cur, 1.0 - t)  # so we don't overshoot the interval
        h_current = jnp.minimum(h_current, self._h_max)
        h_current = jnp.maximum(h_current, self._h_min)
        # jax.debug.print("Performing step")
        k_init = jnp.stack(
            [
                jnp.zeros_like(x),
            ]
            * (self.n_stages + 1),
            axis=0,
        )

        err_treshold = self._rtol * jnp.linalg.norm(x.ravel()) + self._atol

        k_init = k_init.at[0].set(self._rhs(t, x, *args))

        def _cond_trunc_err(carry):
            _, _, _, h_new, trunc_err_normalized = carry
            return (trunc_err_normalized >= 1.0) & (h_new >= self._h_min)

        def _iterate_for_step(carry):
            # jax.debug.print("\tIterating for step")
            A, B, C, E = self.butcher_tableau
            _, _, k, h_cur, trunc_err_normalized = carry
            h_cur = jnp.maximum(h_cur, self._h_min)
            h_cur = jnp.minimum(h_cur, 1.0 - t)

            for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
                # Combine s previous k's with coefficients from A[s, :s]
                # Tensordot needed to handle arbitrary shape of x
                dx = jnp.tensordot(k[:s], a[:s], axes=((0,), (0,))) * h_cur
                k_cur = self._rhs(t + c * h_cur, x + dx, *args)
                # jax.debug.print("\t\t\tEvaluating rhs")
                k = k.at[s].set(k_cur)
                # jax.debug.print("{0}", k)

            x_new = x + h_cur * jnp.tensordot(k[:-1], B, axes=((0,), (0,)))
            # Compute x_estimate by 4-th order formula
            # instead of computing the error as difference between x_4 and x_5
            # use formula |\Sum((c^4_i - c^5_i)k_i)|
            # with E_i := (c^4_i - c^5_i)
            k = k.at[-1].set(self._rhs(t + h_cur, x_new, *args))
            trunc_err = h_cur * jnp.linalg.norm(
                jnp.tensordot(k, E, axes=((0,), (0,))).ravel()
            )

            trunc_err_normalized = trunc_err / err_treshold

            # TODO replace 5 with approximation order (make arbitrary Runge-Kutta?)
            h_new = 0.9 * h_cur * (trunc_err_normalized) ** (-1 / 5)

            return t + h_cur, x_new, k, h_new, trunc_err_normalized

        def _body_fn(carry, i=None):
            cond_value = _cond_trunc_err(carry)
            h_new, trunc_err_normalized = carry[-2:]
            # jax.debug.print("t = {3} i = {0}, te = {1} h_new = {2}", i, trunc_err_normalized, h_new, t)
            return nnx.cond(cond_value, _iterate_for_step, lambda _c: _c, carry)

        t_new, x_new, k, h_new, trunc_err_normalized = nnx.fori_loop(
            0,
            self._N_iter_to_accept,
            lambda i, carry: _body_fn(carry, i=i),
            (t, x.copy(), k_init, h_current, 1.0),
        )

        return t_new, x_new, h_new

    def suggest_h0(
        self,
        h_min: float,
    ):
        return h_min * 10.0

    def tree_flatten(self):
        children = None
        aux_data = dict(
            rhs=self._rhs,
            rtol=self._rtol,
            atol=self._atol,
            h_min=self._h_min,
            h_max=self._h_max,
            N_iter_to_accept=self._N_iter_to_accept,
        )

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


@jax.tree_util.register_pytree_node_class
class EulerStep(ODEStep):
    def __init__(
        self,
        rhs: Callable,
        **kwargs,
    ):
        self._rhs = rhs

    def __call__(self, t: jnp.float32, x: jnp.ndarray, h: float, *args):
        return t + h, x + h * self._rhs(t, x, *args), h

    def tree_flatten(self):
        children = None
        aux_data = dict(
            rhs=self._rhs,
        )

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


@jax.tree_util.register_pytree_node_class
class HeunStep(ODEStep):
    """Heun's method for solving ODEs."""

    def __init__(
        self,
        rhs: Callable,
        **kwargs,
    ):
        self._rhs = rhs

    def __call__(self, t: jnp.float32, x: jnp.ndarray, h: jnp.float32, *args):
        """Perform one Heun integration step."""
        # Predictor step (Euler)
        xdot_cur = self._rhs(t, x, *args)
        x_predictor = x + h * xdot_cur

        # Corrector step
        x_new = x + (h / 2) * (xdot_cur + self._rhs(t + h, x_predictor, *args))

        return t + h, x_new, h

    def tree_flatten(self):
        children = None
        aux_data = dict(
            rhs=self._rhs,
        )

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


_method_name_to_class = dict(rk45=RK45Step, euler=EulerStep, heun=HeunStep)


def solve_ode(
    rhs: Callable,
    x0,
    *aux_args,
    N_steps_max: int = 100,
    method: Literal["rk45", "euler", "heun"] = "rk45",
    **method_kw,
):
    h_min = 1.0 / N_steps_max
    step = _method_name_to_class.get(method)(rhs, **(method_kw | dict(h_min=h_min)))
    h0 = step.suggest_h0(h_min)

    def _body_fn(i, carry):
        t_cur, x_cur, h_new = carry
        # the idea is to iterate over n, and tracking the integrated time t_cur
        # such that t_cur(i) >= i / N for every i (*)
        # at i = 0 it holds that t_cur(0) = 0. = 0 / N;
        # assume for some i > 0 (*) holds
        # j = jax.lax.axis_index("batch")
        # jax.debug.print(
        #     "i: {1} t_cur >= (i + 1) * h_min = {0}",
        #     t_cur >= (i + 1) * h_min,
        #     j
        # )
        return nnx.cond(
            t_cur >= (i + 1) * h_min,
            lambda _c: _c,  # if for the same t_cur (*) also holds for i + 1, do nothing
            lambda _c: step(
                *_c,
                *aux_args,
            ),  # else, do step; since the minimal stepsize is 1 / N, (*) is guaranteed to hold for i + 1
            carry,
        )

    t, x, _ = nnx.fori_loop(0, N_steps_max, _body_fn, (0.0, x0, h0))

    return x


def solve_ode_batched(
    rhs: Callable,
    x0_batched,
    *aux_args_batched,
    N_steps_max: int = 100,
    method: Literal["rk45", "euler", "heun"] = "rk45",
    **method_kw,
):
    """Batched ODE solver that avoids vmap overhead by implementing proper batched logic.

    This version uses a simpler approach without the complex conditional logic
    that requires batch axis indexing, avoiding the performance issues with vmap.

    Args:
        rhs: Right-hand side function of the ODE
        x0_batched: Batch of initial conditions with shape (B, ...)
        N_steps_max: Maximum number of integration steps
        method: Integration method to use
        **method_kw: Additional keyword arguments for the method

    Returns:
        Batch of solutions at t=1.0 with shape (B, ...)
    """
    batch_size = x0_batched.shape[0]
    h_min = 1.0 / N_steps_max
    step = _method_name_to_class.get(method)(rhs, **(method_kw | dict(h_min=h_min)))
    step_batched = jax.vmap(step)
    h0 = step.suggest_h0(h_min)

    # Initialize batched state
    t_batch = jnp.zeros((batch_size,))
    x_batch = x0_batched
    h_batch = jnp.full((batch_size,), h0)

    def _body_fn(i, carry):
        t_batch, x_batch, h_batch = carry[:3]

        # Target time for this iteration
        target_time = (i + 1) * h_min
        target_time = jnp.minimum(target_time, 1.0)

        # Only step if we haven't reached the target time
        needs_step = t_batch < target_time

        carry_new = jax.lax.cond(
            jnp.any(needs_step),
            lambda _c: step_batched(*_c) + aux_args_batched,
            lambda _c: _c,
            carry,
        )

        # Ensure we don't overshoot t=1.0
        t_new = carry_new[0]
        t_new = jnp.minimum(t_new, 1.0)

        return t_new, *carry_new[1:]

    carry = nnx.fori_loop(
        0, N_steps_max, _body_fn, (t_batch, x_batch, h_batch, *aux_args_batched)
    )
    x_final = carry[1]

    return x_final
