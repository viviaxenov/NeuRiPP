"""Explicit, one-step optimizer check; not collected by pytest.

Run in separate processes: python tests/smoke_pde_optimizer.py ks adam
or acoustic ngd. Optional --size selects a grid resolution.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=("ks", "acoustic"))
    parser.add_argument("method", choices=("adam", "ngd"))
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from logpdf_targets import get_logpdf_ks_inverse_problem, get_logpdf_acoustic_inverse_problem
    from rhs_ks import KSDilatedResNetRHS
    from rhs_acoustic import AcousticUNetRHS
    from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
    from neuripp.functionals.KL import getKL
    from neuripp.methods.ngd import get_ngd
    from neuripp.methods.optax_optimizer import get_optax

    rngs = nnx.Rngs(42)
    if args.problem == "ks":
        target = get_logpdf_ks_inverse_problem(args.size, min(8, args.size))
        rhs = KSDilatedResNetRHS(args.size, rngs=rngs)
    else:
        target = get_logpdf_acoustic_inverse_problem(args.size, 4, 8)
        rhs = AcousticUNetRHS(args.size**2, rngs=rngs)
    model = ParametricPushforward(rhs, rngs, 1, ode_nstep_max=2,
                                 ode_method="euler", divergence_method="hutchinson")
    loss = getKL(target)
    before = [np.array(x) for x in jax.tree.leaves(nnx.state(model, nnx.Param))]
    batch = jax.random.normal(jax.random.PRNGKey(7), (1, rhs.dim))
    if args.method == "adam":
        initialize, step = get_optax(loss, "adam")
        state = initialize(model, (1e-4,), {})
    else:
        initialize, step = get_ngd(loss, natural_grad_clipping_threshold=1.)
        state = initialize(model, (1e-4,), dict(linear_solver_regularization=0.1,
                                              linear_solver_maxiter=3))
    state, metrics = nnx.jit(step)(state, batch, rngs)
    jax.block_until_ready(metrics)
    after = [np.asarray(x) for x in jax.tree.leaves(nnx.state(state[0], nnx.Param))]
    assert all(np.all(np.isfinite(x)) for x in after)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(metrics))
    delta = sum(float(np.sum((x - y)**2)) for x, y in zip(before, after))**0.5
    assert delta > 0, "Optimizer did not update parameters"
    print(f"{args.problem} {args.method}: loss={float(metrics[0]):.8g}, parameter_delta_l2={delta:.8g}")


if __name__ == "__main__":
    main()
