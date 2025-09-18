import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Callable, Optional, Any
from jaxtyping import PyTree, Array
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jax import lax


def reg_cg(
    A_func: Callable,
    b: PyTree,
    epsilon: float = 1e-6,
    tol: float = 1e-6,
    x0: Optional[PyTree] = None,
    maxiter: int = 100,
) -> tuple[PyTree, dict]:

    def regu_A(x: PyTree) -> PyTree:
        return jax.tree.map(lambda x, y: x + epsilon * y, A_func(x), x)

    return cg(regu_A, b, x0=x0, tol=tol, maxiter=maxiter)


def minres(
    A_func: Callable,
    b: PyTree,
    tol: float = 1e-6,
    x0: Optional[PyTree] = None,
    maxiter: int = 100,
) -> tuple[PyTree, dict]:
    """
    MINRES implementation for PyTree support.
    """

    @jax.jit
    def dot_tree(x: PyTree, y: PyTree) -> Array:
        return sum(jax.tree.leaves(jax.tree.map(lambda a, b: jnp.sum(a * b), x, y)))

    @jax.jit
    def norm_tree(x: PyTree) -> Array:
        return jnp.sqrt(dot_tree(x, x))

    @jax.jit
    def scale_tree(x: PyTree, alpha: float) -> PyTree:
        return jax.tree.map(lambda a: alpha * a, x)

    @jax.jit
    def add_trees(x: PyTree, y: PyTree) -> PyTree:
        return jax.tree.map(lambda a, b: a + b, x, y)

    @jax.jit
    def sub_trees(x: PyTree, y: PyTree) -> PyTree:
        return jax.tree.map(lambda a, b: a - b, x, y)

    # Initialize
    if x0 is None:
        x = jax.tree.map(jnp.zeros_like, b)
    else:
        x = jax.tree.map(lambda a: jnp.array(a), x0)

    # Initial residual
    Ax = A_func(x)
    r = sub_trees(b, Ax)

    # MINRES initialization
    v_old = jax.tree.map(jnp.zeros_like, b)
    v = jax.tree.map(lambda a: jnp.array(a), r)
    w_old = jax.tree.map(jnp.zeros_like, b)
    w = jax.tree.map(jnp.zeros_like, b)

    beta = norm_tree(r)
    initial_residual = beta

    if beta < tol:
        info = {"success": True, "iterations": 0, "norm_res": beta}
        return x, info

    # Normalize v
    v = scale_tree(v, 1.0 / beta)

    eta = beta
    s_old = 0.0
    c_old = 1.0

    for i in range(maxiter):
        # Lanczos process
        Av = A_func(v)

        if i > 0:
            Av = sub_trees(Av, scale_tree(v_old, beta))

        alpha = dot_tree(v, Av)
        Av = sub_trees(Av, scale_tree(v, alpha))

        beta = norm_tree(Av)

        # Apply previous Givens rotation
        if i > 0:
            delta = c_old * alpha + s_old * beta
            gamma_bar = s_old * alpha - c_old * beta
        else:
            delta = alpha
            gamma_bar = beta

        # Compute new Givens rotation
        if abs(gamma_bar) < 1e-14:  # Avoid division by zero
            c = 1.0
            s = 0.0
            gamma = delta
        else:
            if abs(gamma_bar) > abs(delta):
                tau = delta / gamma_bar
                s = 1.0 / jnp.sqrt(1.0 + tau**2)
                c = s * tau
            else:
                tau = gamma_bar / delta
                c = 1.0 / jnp.sqrt(1.0 + tau**2)
                s = c * tau
            gamma = c * delta + s * gamma_bar

        # Update solution
        if abs(gamma) > 1e-14:  # Avoid division by zero
            eta_new = -s * eta
            eta = c * eta

            w_new = sub_trees(
                sub_trees(v, scale_tree(w_old, gamma_bar)), scale_tree(w, delta)
            )
            w_new = scale_tree(w_new, 1.0 / gamma)

            x = add_trees(x, scale_tree(w_new, eta))

            # Update for next iteration
            w_old = jax.tree.map(lambda a: jnp.array(a), w)
            w = jax.tree.map(lambda a: jnp.array(a), w_new)
            eta = eta_new

        # Check convergence
        residual_norm = abs(eta)

        if residual_norm < tol:
            info = {"success": True, "iterations": i + 1, "norm_res": residual_norm}
            return x, info

        # Prepare for next iteration
        if beta > 1e-14 and i < maxiter - 1:
            v_old = jax.tree.map(lambda a: jnp.array(a), v)
            v = scale_tree(Av, 1.0 / beta)
            c_old = c
            s_old = s
        else:
            break

    # Did not converge
    info = {"success": False, "iterations": maxiter, "norm_res": residual_norm}
    return x, info
