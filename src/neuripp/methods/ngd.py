import jax
import jax.numpy as jnp
import jax.scipy as jsp

from typing import Callable

from flax import nnx
from neuripp.utility.utility import tree_dot_product


def _clip_gradient(natural_grad, norm: float, max_norm: float):
    factor = jnp.minimum(1.0, max_norm / norm)
    return jax.tree.map(lambda _x: _x * factor, natural_grad)

def schedule_exp(step_size: float, iter_count: int, drop_every: int = 100, drop_by: float = 1.1, min_step: float = 0., max_step:float= 1.0, **kwargs):
    predicate = ((iter_count + 1) % drop_every == 0)
    new_step = jax.lax.cond(predicate, lambda _x: jnp.clip(_x / drop_by, min_step, max_step), lambda _x: _x, step_size)

    return new_step


def schedule_polynomial(
    peak_step: float,
    iter_count: int,
    *,
    min_lr: float = 1e-8,
    iterations: int,
    warmup_steps: int = 0,
    p: float = 1.0,
    **kwargs,
):
    """Linear warmup followed by polynomial decay to ``min_lr``.

    ``iter_count`` is zero-based.  The peak is reached at ``warmup_steps``
    and the floor at ``iterations``; values beyond the horizon stay at the
    floor.  The peak is passed separately so NGD scheduling never compounds
    decay through its mutable optimizer argument.
    """
    del kwargs
    step = jnp.asarray(iter_count, dtype=jnp.result_type(peak_step, min_lr))
    peak = jnp.asarray(peak_step)
    floor = jnp.asarray(min_lr)
    warmup = jnp.asarray(warmup_steps, dtype=step.dtype)
    horizon = jnp.asarray(iterations, dtype=step.dtype)
    warmup_fraction = jnp.clip(step / jnp.maximum(warmup, 1), 0.0, 1.0)
    decay_fraction = jnp.clip(
        (step - warmup) / jnp.maximum(horizon - warmup, 1), 0.0, 1.0
    )
    warmup_value = floor + (peak - floor) * warmup_fraction
    decay_value = floor + (peak - floor) * (1.0 - decay_fraction) ** p
    return jax.lax.select(step < warmup, warmup_value, decay_value)



def _compute_natural_grad(
    model,
    rngs,
    grad,
    init_vector,
    data_batch=None,
    linear_solver_regularization: float = 1e-3,
    linear_solver_tolerance: float = 1e-6,
    linear_solver_maxiter: float = 50,
    **kwargs,
):
    matvec_cur = model.get_matvec_fn(rngs, data_batch=data_batch)

    def oper_cg(tang):
        Gtang = matvec_cur(tang)
        return jax.tree.map(
            lambda _g, _x: _g + linear_solver_regularization * _x,
            Gtang,
            tang,
        )

    natural_grad = jsp.sparse.linalg.cg(
        oper_cg,
        grad,
        init_vector,
        tol=linear_solver_tolerance,
        maxiter=linear_solver_maxiter,
    )[0]

    return natural_grad


def get_ngd(
    loss: Callable,
    linear_solver_method: str = "cg",
    natural_grad_clipping_threshold: float = None,
    stepsize_schedule_fn: Callable = None,
):
    """
    Gives `jax.lax.scan`-compatible function for the Picard/NGD method
    """
    if linear_solver_method != "cg":
        raise ValueError(
            f"Only linear_solver_method='cg' is implemented so far, but got {linear_solver_method}"
        )
    vg_fn = nnx.value_and_grad(loss, argnums=0)

    def _ngd_init(model, args, kwargs, *rest_args):
        _, par, _ = nnx.split(model, nnx.Param, ...)
        previous_grad = jax.tree.map(jnp.zeros_like, par)
        state = (model, previous_grad, 0, args, kwargs)
        return state

    def _ngd_step(
        state,
        batch,
        rngs,
        matvec_batch=None,
    ):
        model, prev_grad, i, args, kwargs = state
        step_size = args[0]

        if stepsize_schedule_fn is not None:
            step_size = stepsize_schedule_fn(step_size, i, **kwargs)

        # Compute the gradient in training mode, then the metric in eval mode.
        model.train()
        f, grad = vg_fn(model, batch, rngs)

        model.eval()
        # compute natural grad
        # natural_grad_norm_alt = tree_dot_product(grad, natural_grad)
        natural_grad = _compute_natural_grad(
            model,
            rngs,
            grad,
            prev_grad,
            data_batch=batch if matvec_batch is None else matvec_batch,
            **kwargs,
        )
        grad_norm_sq = tree_dot_product(grad, grad)

        natural_grad_norm_sq = model.scalar_product(
            natural_grad,
            natural_grad,
            rngs,
            data_batch=batch if matvec_batch is None else matvec_batch,
        )
        norm = jnp.maximum(natural_grad_norm_sq, 0.0) ** 0.5
        # Gradient clipping
        if natural_grad_clipping_threshold is not None:
            natural_grad = _clip_gradient(
                natural_grad, norm, natural_grad_clipping_threshold
            )

        gd, params, rest = nnx.split(model, nnx.Param, ...)
        # update params
        params_new = jax.tree.map(lambda x, y: x - y * step_size, params, natural_grad)
        model = nnx.merge(gd, params_new, rest)
        model.train()
        args = (step_size, *args[1:])

        return (model, natural_grad, i+ 1, args, kwargs), (
            f,
            grad_norm_sq,
            natural_grad_norm_sq,
        )

    return _ngd_init, _ngd_step
