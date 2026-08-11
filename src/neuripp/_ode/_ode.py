import abc
from typing import Callable, Tuple, Literal
import jax
import jax.numpy as jnp
import jax.ad_checkpoint
from flax import nnx


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
        N_iter_to_accept: int = 10,
    ):
        self._rhs = rhs
        self._rtol = rtol
        self._atol = atol
        self._h_min = h_min
        self._h_max = h_max
        self._N_iter_to_accept = N_iter_to_accept

    def check_step(self, h: jnp.float32, t: jnp.float32):
        h_verif = jnp.clip(h, self._h_min, self._h_max)
        h_verif = jnp.minimum(h_verif, 1.0 - t)  # so we don't overshoot the interval
        h_verif = jnp.where(
            h <= 0.0, 0.0, h_verif
        )  # h == 0. can be come from the batched solve when this element of the batch has already finished

        return h_verif

    def iterate_for_step(
        self,
        t: jnp.float32,
        x: jnp.ndarray,
        h_cur: jnp.float32,
        *args,
    ):
        def cond_fn(carry):
            h, trunc_err_normalized, i = carry
            step_size_ok = jnp.logical_and(h >= self._h_min, h <= self._h_max)
            return jnp.logical_and(
                i < self._N_iter_to_accept,
                jnp.logical_and(step_size_ok, trunc_err_normalized >= 1.0),
            )

        def body_fn(carry):
            h, trunc_err_normalized, i = carry
            h_new, ten_new = self(t, x, h, *args, return_err=True)
            return h_new, ten_new, i + 1

        h_suggested, _, _ = nnx.while_loop(cond_fn, body_fn, (h_cur, 1.0, 0))
        return self.check_step(h_suggested, t)

    def __call__(
        self,
        t: jnp.float32,
        x: jnp.ndarray,
        h_cur: jnp.float32,
        *args,
        return_err=False,
    ):
        A, B, C, E = self.butcher_tableau

        k0 = self._rhs(t, x, *args)  # TODO: k0 caching for iteration
        k1 = self._rhs(t + C[1] * h_cur, x + A[1, 0] * k0 * h_cur, *args)
        k2 = self._rhs(
            t + C[2] * h_cur,
            x + (A[2, 0] * k0 + A[2, 1] * k1) * h_cur,
            *args,
        )
        k3 = self._rhs(
            t + C[3] * h_cur,
            x + (A[3, 0] * k0 + A[3, 1] * k1 + A[3, 2] * k2) * h_cur,
            *args,
        )
        k4 = self._rhs(
            t + C[4] * h_cur,
            x + (A[4, 0] * k0 + A[4, 1] * k1 + A[4, 2] * k2 + A[4, 3] * k3) * h_cur,
            *args,
        )
        k5 = self._rhs(
            t + C[5] * h_cur,
            x
            + (A[5, 0] * k0 + A[5, 1] * k1 + A[5, 2] * k2 + A[5, 3] * k3 + A[5, 4] * k4)
            * h_cur,
            *args,
        )

        x_new = x + h_cur * (
            B[0] * k0 + B[1] * k1 + B[2] * k2 + B[3] * k3 + B[4] * k4 + B[5] * k5
        )
        if return_err:
            # Compute x_estimate by 4-th order formula
            # instead of computing the error as difference between x_4 and x_5
            # use formula |\Sum((c^4_i - c^5_i)k_i)|
            # with E_i := (c^4_i - c^5_i)
            k6 = self._rhs(t + h_cur, x_new, *args)
            trunc_err = h_cur * jnp.linalg.norm(
                (
                    E[0] * k0
                    + E[1] * k1
                    + E[2] * k2
                    + E[3] * k3
                    + E[4] * k4
                    + E[5] * k5
                    + E[6] * k6
                ).ravel()
            )
            err_threshold = self._rtol * jnp.linalg.norm(x.ravel()) + self._atol
            trunc_err_normalized = trunc_err / err_threshold
            trunc_err_normalized = jnp.maximum(trunc_err_normalized, 1e-25)
            h_new = 0.9 * h_cur * (trunc_err_normalized) ** (-1 / 5)
            return h_new, trunc_err_normalized

        return x_new

    def suggest_h0(
        self,
        h_min: float,
    ):
        return (h_min + self._h_max) / 2.0

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
        return x + h * self._rhs(t, x, *args)

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

        return x_new

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
    adaptive: bool = False,
    grad_checkpointing: Literal[None, "dots_only", "save", "offload"] = None,
    grad_checkpoint_every: int = 2,
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
    if grad_checkpointing in []:
        raise NotImplementedError(
            f"Checkpointing strategy {grad_checkpointing} not supported yet"
        )

    batch_size = x0_batched.shape[0]
    h_min = 1.0 / N_steps_max
    step = _method_name_to_class.get(method)(rhs, **(method_kw | dict(h_min=h_min)))
    step_batched = jax.vmap(step)
    h0 = step.suggest_h0(h_min)

    # Initialize batched state
    x_batch = x0_batched
    t_batch = jnp.zeros((batch_size,))
    h_batch = jnp.full((batch_size,), h0)

    if adaptive:
        get_step_batched = jax.vmap(step.iterate_for_step)

        def _do_step_fn(carry):
            t, x, h = carry
            h_new = jax.lax.stop_gradient(get_step_batched(t, x, h, *aux_args_batched))

            t_new = t + h_new
            x_new = step_batched(t, x, h_new, *aux_args_batched)

            return t_new, x_new, h_new

        @nnx.remat
        def _skip_fn(carry):
            return carry

        if grad_checkpointing is not None:
            _do_step_fn = nnx.remat(
                _do_step_fn,
                prevent_cse=False,
                policy=jax.ad_checkpoint.checkpoint_policies.dots_with_no_batch_dims_saveable,
            )

        def _body_fn(i, carry):
            t_batch, x_batch, h_batch = carry

            # Only step if we haven't reached the target time
            needs_step = t_batch < 1.0

            t_new, x_new, h_new = nnx.cond(
                jnp.any(needs_step),
                _do_step_fn,
                _skip_fn,
                carry,
            )

            batch_mask = needs_step.reshape(
                (needs_step.shape[0],) + (1,) * (x_batch.ndim - 1)
            )
            x_new = jnp.where(batch_mask, x_new, x_batch)

            return t_new, x_new, h_new

    else:
        times = jnp.broadcast_to(
            jnp.linspace(0.0, 1.0, N_steps_max)[:, jnp.newaxis],
            (N_steps_max, batch_size),
        )
        steps = jnp.full_like(times, h_min)

        def _body_fn(i, carry):
            t, x, h = carry
            return t + h, step_batched(t, x, h, *aux_args_batched), h

        if grad_checkpointing is not None:
            _body_fn = nnx.remat(
                _body_fn,
                prevent_cse=False,
                policy=jax.ad_checkpoint.checkpoint_policies.dots_with_no_batch_dims_saveable,
            )

    carry = nnx.fori_loop(0, N_steps_max, _body_fn, (t_batch, x_batch, h_batch))
    x_final = carry[1]

    return x_final
