import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Union
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from flax import nnx
import jax

from architectures.node import NeuralODE
from functionals.linear_funcitonal_class import LinearPotential as LinearFunctional
from functionals.internal_functional_class import (
    InternalPotential as InternalFunctional,
)
from functionals.interaction_functional_class import (
    InteractionPotential as InteractionFunctional,
)


class Potential:
    """
    A class to manage the three potentials: linear, internal, interaction
    #TODO Implement interaction potential
    1. Linear potential: F(ρ) = ∫ U(x)ρ(x)dx
    2. Internal potential: F(ρ) = ∫ f(ρ(x))dx
    3. Interaction potential: F(ρ) = 0.5 ∫∫ W(x-y)ρ(x)ρ(y)dxdy
    where U is a user-defined potential function, f is a user-defined function, and W is a user-defined interaction function.
    The current implementation only support the entropy and Fisher information for the internal potential.
    These quantities are computed by solving ODEs, and their computation is done in the node.py file.
    """

    def __init__(
        self,
        linear: LinearFunctional,
        internal: InternalFunctional,
        interaction: InteractionFunctional,
    ):
        self.linear = linear
        self.internal = internal
        self.interaction = interaction

    def evaluate_energy(
        self, node: NeuralODE, z_samples: Array, params: Optional[PyTree] = None
    ) -> float:
        """
        Evaluate the total energy functional F(ρ) = F_linear(ρ) + F_internal(ρ) + F_interaction(ρ)
        Args:
            node: Neural ODE model
            z_samples: Reference samples (batch_size, d)
            params: Optional PyTree of parameters for the dynamics model
        Returns:
            energy: Total energy functional
        """

        if params is None:
            _, params = nnx.split(node)

        # Transform reference samples
        z_trajectory, time_steps = node(
            z_samples, history=True, params=params
        )  # (batch_size, time_steps, dim)
        x_samples = z_trajectory[:, -1, :]
        energy = 0.0
        linear_energy = 0.0
        internal_energy = 0.0
        interaction_energy = 0.0
        # Linear potential
        if self.linear is not None:
            linear_energy, _ = self.linear.evaluate_energy(
                node, z_samples, x_samples=x_samples
            )
            linear_energy = linear_energy * self.linear.coeff
            energy += linear_energy
        # Internal potential
        if self.internal is not None:
            internal_energy = self.internal(
                node=node,
                z_samples=z_samples,
                z_trajectory=z_trajectory,
                time_steps=time_steps,
                params=params,
            )
            energy += internal_energy
        # Interaction potential
        # TODO

        return energy, x_samples, linear_energy, internal_energy, interaction_energy

    def compute_energy_gradient(
        self, node: NeuralODE, z_samples: Array, params: PyTree
    ) -> PyTree:
        """
        Compute the gradient of the total energy functional F(ρ) = F_linear(ρ) + F_internal(ρ) + F_interaction(ρ)
        Args:
            node: Neural ODE model
            z_samples: Reference samples (batch_size, d)
            params: Optional PyTree of parameters for the dynamics model
        Returns:
            energy_gradient: Gradient of the total energy functional
        """

        if params is None:
            _, params = nnx.split(node)

        def energy_evaluation(p: PyTree) -> Array:
            # Transform reference samples
            z_trajectory, time_steps = node(
                z_samples, history=True, params=p
            )  # (batch_size, time_steps, dim)
            x_samples = z_trajectory[:, -1, :]
            energy = 0.0
            linear_energy = 0.0
            internal_energy = 0.0
            interaction_energy = 0.0
            # Linear potential
            if self.linear is not None:

                linear_energy, _ = self.linear.evaluate_energy(
                    node, z_samples, x_samples=x_samples
                )
                linear_energy = linear_energy * self.linear.coeff
                energy += linear_energy
            # Internal potential
            if self.internal is not None:
                internal_energy = self.internal(
                    node=node,
                    z_samples=z_samples,
                    z_trajectory=z_trajectory,
                    time_steps=time_steps,
                    params=p,
                )

                energy += internal_energy
            # Interaction potential
            # TODO

            energy_breakdown = {
                "internal_energy": internal_energy,
                "linear_energy": linear_energy,
                "interaction_energy": interaction_energy,
            }
            return energy, energy_breakdown

        (energy, energy_breakdown), energy_grad = jax.value_and_grad(
            energy_evaluation, has_aux=True
        )(params)

        return energy_grad, energy, energy_breakdown
