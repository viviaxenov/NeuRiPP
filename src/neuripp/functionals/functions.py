from typing import Callable, Optional, Any
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from flax import nnx

from .linear_funcitonal_class import LinearPotential


# Define specific potential functions
@jax.jit
def quadratic_potential_fn(x: Array) -> Array:
    """
    Quadratic potential U(x) = |x|²/2
    """
    return jnp.sum(x**2, axis=-1) / 2.0


@jax.jit
def double_well_potential_fn(x: Array, alpha: float = 1.0) -> Array:
    """
    Double-well potential U(x,y) = (y² - 1)² + α*x²
    """
    x_coord = x[:, 0]  # x-coordinates
    y_coord = x[:, 1]  # y-coordinates

    # Double-well in y: (y² - 2)²
    y_term = (y_coord**2 - 2.0) ** 2

    # Harmonic confinement in x: αx²
    x_term = alpha * x_coord**2

    return y_term + x_term


@jax.jit
def four_well_potential_fn(x: Array, alpha: float = 1.0) -> Array:
    """
    Four-well potential U(x,y) = (y² - 2)² + α*(x²-2)²
    """
    x_coord = x[:, 0]  # x-coordinates
    y_coord = x[:, 1]  # y-coordinates

    # Four-well in y: (y² - 2)²
    y_term = alpha * (y_coord**2 - 2.0) ** 2

    # Harmonic confinement in x: α*(x² - 2)²
    x_term = alpha * (x_coord**2 - 2.0) ** 2

    return y_term + x_term


@jax.jit
def quartic_potential_fn(x: Array, strength: float = 0.1) -> Array:
    """
    Quartic potential U(x) = strength * |x|⁴ + 0.5 * |x|²
    """
    r_squared = jnp.sum(x**2, axis=-1)
    return strength * r_squared**2 + 0.5 * r_squared


@jax.jit
def styblinski_tang_potential_fn(x: Array, d: int = 2) -> Array:
    """
    Styblinski-Tang potential U(x) = 0.5 * Σ (x_i^4 - 16*x_i^2 + 5*x_i)
    Global minimum at x_i = -2.903534 for all i
    """
    return 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x, axis=-1)


# Create different potential instances
def create_potentials():
    """Factory function to create common potential instances"""

    # Quadratic potential (particles move to origin)
    quadratic_potential = LinearPotential(quadratic_potential_fn)

    # Double-well potential (particles move to [0,±1])
    double_well_potential = LinearPotential(double_well_potential_fn, alpha=1.0)

    # Strong double-well potential (narrower wells)
    strong_double_well = LinearPotential(double_well_potential_fn, alpha=5.0)

    # Four-well potential
    four_well_potential = LinearPotential(four_well_potential_fn, alpha=1.0)

    # Strong four-well potential
    strong_four_well = LinearPotential(four_well_potential_fn, alpha=5.0)
    # Quartic potential (softer confinement)
    quartic_potential = LinearPotential(quartic_potential_fn, strength=0.1)
    # Styblinski-Tang potential
    styblinski_tang_potential = LinearPotential(styblinski_tang_potential_fn, d=2)

    return {
        "quadratic": quadratic_potential,
        "double_well": double_well_potential,
        "strong_double_well": strong_double_well,
        "four_well": four_well_potential,
        "strong_four_well": strong_four_well,
        "quartic": quartic_potential,
        "styblinski_tang": styblinski_tang_potential,
    }
