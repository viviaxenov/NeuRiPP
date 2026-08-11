"""Adapters for the project's existing optimization-method factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from neuripp.methods.anderson import get_anderson
from neuripp.methods.ngd import get_ngd, schedule_exp
from neuripp.methods.optax_optimizer import get_optax, optax_optimizers


CUSTOM_METHODS = {"ngd", "anderson"}
UNSUPPORTED_OPTAX_METHODS = {"lbfgs"}
METHOD_NAMES = CUSTOM_METHODS | (set(optax_optimizers) - UNSUPPORTED_OPTAX_METHODS)


@dataclass(frozen=True)
class ResolvedMethod:
    name: str
    init_fn: Callable
    step_fn: Callable
    optimizer_args: tuple[Any, ...]
    optimizer_kwargs: dict[str, Any]
    initialization_updates: int = 0


def _schedule(kwargs: dict[str, Any]):
    name = kwargs.pop("stepsize_schedule", kwargs.pop("stepsize_schedule_name", None))
    if name is None:
        return None
    if name != "schedule_exp":
        raise ValueError("Only stepsize_schedule='schedule_exp' is supported")
    return schedule_exp


def resolve_method(
    config: dict[str, Any], loss: Callable
) -> ResolvedMethod:
    """Resolve config without introducing optimizer logic elsewhere."""

    name = config.get("name")
    if name not in METHOD_NAMES:
        supported = ", ".join(sorted(METHOD_NAMES))
        raise ValueError(f"Unknown optimizer method {name!r}; expected: {supported}")
    kwargs = dict(config.get("kwargs", {}))
    if name in optax_optimizers:
        if "beta1" in kwargs:
            if "b1" in kwargs:
                raise ValueError("Specify only one of beta1 and b1")
            kwargs["b1"] = kwargs.pop("beta1")
        if "beta2" in kwargs:
            if "b2" in kwargs:
                raise ValueError("Specify only one of beta2 and b2")
            kwargs["b2"] = kwargs.pop("beta2")
        learning_rate = kwargs.pop("learning_rate", kwargs.pop("step_size", None))
        if learning_rate is None:
            raise ValueError(f"Optimizer {name!r} requires learning_rate")
        init_fn, step_fn = get_optax(loss, method=name)
        return ResolvedMethod(name, init_fn, step_fn, (learning_rate,), kwargs)

    schedule = _schedule(kwargs)
    factory_kwargs: dict[str, Any] = {}
    for key in ("linear_solver_method", "natural_grad_clipping_threshold"):
        if key in kwargs:
            factory_kwargs[key] = kwargs.pop(key)
    if schedule is not None:
        factory_kwargs["stepsize_schedule_fn"] = schedule
    step_size = kwargs.pop("step_size", None)
    if step_size is None:
        raise ValueError(f"Optimizer {name!r} requires step_size")

    if name == "ngd":
        init_fn, step_fn = get_ngd(loss, **factory_kwargs)
        return ResolvedMethod(name, init_fn, step_fn, (step_size,), kwargs)

    for key in ("history_length", "regularization_method", "ensure_descent"):
        if key in kwargs:
            factory_kwargs[key] = kwargs.pop(key)
    relaxation = kwargs.pop("relaxation", 1.0)
    regularization_factor = kwargs.pop(
        "regularization_factor", kwargs.pop("reg_factor", None)
    )
    if regularization_factor is None:
        raise ValueError("Anderson requires regularization_factor")
    init_fn, step_fn = get_anderson(loss, **factory_kwargs)
    return ResolvedMethod(
        name,
        init_fn,
        step_fn,
        (step_size, relaxation, regularization_factor),
        kwargs,
        initialization_updates=1,
    )
