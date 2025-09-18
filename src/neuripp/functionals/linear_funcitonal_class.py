from typing import Callable, Optional, Any
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt


class LinearPotential:
    """
    A class for handling linear potential energy functionals F(ρ) = ∫ U(x)ρ(x)dx
    where U(x) is a user-defined potential function.
    """

    def __init__(
        self,
        potential_fn: Callable[[Array], Array],
        coeff: Optional[float] = 0.0,
        x_bds: Optional[Array] = None,
        y_bds: Optional[Array] = None,
        **potential_kwargs
    ):
        """
        Initialize LinearPotential with a potential function.

        Args:
            potential_fn: Function that takes positions (batch_size, d) and returns
                         potential values (batch_size,)
            **potential_kwargs: Additional keyword arguments for the potential function
        """
        self.potential_fn = potential_fn
        self.potential_kwargs = potential_kwargs
        self.coeff = coeff

        if x_bds is None:
            x_bds = jnp.array([-3, 3])

        if y_bds is None:
            y_bds = jnp.array([-3, 3])

        self.x_bds = x_bds
        self.y_bds = y_bds

    def __call__(self, x: Array) -> Array:
        """
        Evaluate potential at given positions.

        Args:
            x: Positions array of shape (batch_size, d)
        Returns:
            Potential values of shape (batch_size,)
        """
        return self.potential_fn(x, **self.potential_kwargs)

    def compute_energy_gradient(
        self, node: nnx.Module, z_samples: Array, params: Optional[PyTree] = None
    ) -> PyTree:
        """
        Compute gradient of energy functional

        Args:
            node: Neural ODE model
            z_samples: Reference samples (batch_size, d)
            params: Parameters to evaluate gradient at (if None, uses current node params)
        Returns:
            Gradient
        """
        if params is None:
            _, params = nnx.split(node)

        def energy_functional(p: PyTree) -> Array:
            # Transform reference samples through flow
            x_samples = node(z_samples, params=p)
            # Evaluate potential and average (Monte Carlo estimate)
            potential_values = self.potential_fn(x_samples, **self.potential_kwargs)
            return jnp.mean(potential_values)

        values, grad = jax.value_and_grad(energy_functional)(params)

        return grad, values

    def evaluate_energy(
        self,
        node: nnx.Module,
        z_samples: Array,
        x_samples: Optional[Array] = None,
        params: Optional[PyTree] = None,
    ) -> tuple[Array, Array]:
        """
        Evaluate current energy F(ρ_θ)

        Args:
            node: Neural ODE model
            z_samples: Reference samples
            x_samples: Pushforward samples (if None, will compute them)
            params: Parameters (if None, uses current node params)
        Returns:
            (energy_value, transformed_samples)
        """
        return_terminal = False
        if x_samples is None:
            return_terminal = True
            if params is None:
                _, params = nnx.split(node)
            x_samples = node(z_samples, params=params)
        potential_values = self.potential_fn(x_samples, **self.potential_kwargs)
        if return_terminal:
            return jnp.mean(potential_values), x_samples
        else:
            return jnp.mean(potential_values), None

    def plot_function(self, fig=None, ax=None, x_bds=None, y_bds=None):
        """
        Plot the potential function U(x) over the defined boundaries.
        """
        if ax is None:
            fig, ax = plt.subplots()
        if x_bds is None:
            x_bds = self.x_bds
        if y_bds is None:
            y_bds = self.y_bds
        x = jnp.linspace(x_bds[0], x_bds[1], 100)
        y = jnp.linspace(y_bds[0], y_bds[1], 100)
        X, Y = jnp.meshgrid(x, y)
        Z = self.potential_fn(
            jnp.stack([X.ravel(), Y.ravel()], axis=-1), **self.potential_kwargs
        )
        Z = Z.reshape(X.shape)

        contour = ax.contourf(X, Y, Z, levels=100, cmap="cividis", alpha=0.5)
        fig.colorbar(contour)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Potential Function U(x, y)")
        return ax
