import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union, Callable
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx
import jax.scipy.stats as stats

from architectures.node import NeuralODE


class InteractionPotential:
    """
    A class for handling interaction energy functionals F(ρ) = 0.5 ∫∫ W(x-y)ρ(x)ρ(y)dxdy
    where W is a user-defined interaction function.
    """

    def __init__(
        self,
        interaction_fn: Union[str, Callable[[Array, Array], Array]],
        coeff: float = 1.0,
        **interaction_kwargs
    ):
        """
        Initialize InteractionPotential with an interaction function.

        Args:
            interaction_fn: Function that takes positions (batch_size, d) and (batch_size, d) and returns
                         interaction values (batch_size, batch_size). Alternatively, can be 'gaussian' or 'coulomb'.
            coeff: Coefficient for the functional
            **interaction_kwargs: Additional keyword arguments for the interaction function
        """
        # TODO
        return
