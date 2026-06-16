from typing import Callable, Union, Generator, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial

from ..problems.problem import Problem
from ..parametric_pushforward.parametric_pushforward import ParametricPushforward


def get_anderson(
    step_size: float,
    history_length: int,
    relaxation: float,
    regularization_factor: bool,
    regularization_method: Literal['l2', 'adaptive'],
    ensure_descent: bool,
    linear_solver_regularization: float, 
    linear_solver_tolerance: float,
    linear_solver_maxiter: float,
    linear_solver_method: str = "cg",
):

    def _init(
        model: ParametricPushforward,
    ):
        pass

    def _step(model, history):
        pass
