import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import traceback
import argparse
import jax
import jax.numpy as jnp
from jax._src import ad_util
from flax import nnx
import functools
from neuripp._ode._ode import *
from test_rhs import LinearRHS


def batch_rel_err(x1: jnp.ndarray, x2: jnp.ndarray):
    diff = x1 - x2
    denom = 0.5*jnp.abs(x1 + x2)
    denom_norm = jnp.linalg.norm(denom, axis=-1)
    denom_norm = jnp.maximum(denom_norm, 1e-10)

    rel_err = jnp.linalg.norm(diff, axis=-1) / denom_norm

    # return jnp.mean(jnp.max(rel_err, axis=-1))
    return jnp.mean(rel_err)


def _block_tree(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    arr = jnp.array(values)
    return float(jnp.mean(arr)), float(jnp.std(arr))


def test_batched_solver_accuracy(
    batch_size=4,
    dim=3,
    N_steps=50,
    seed=42,
    method="rk45",
    tolerance=1e-2,
    n_restart=10,
    shape=None,
):
    """Test a: batched computation accuracy against reference solution."""
    print(
        f"\n=== Testing Batched Solver Accuracy (batch_size={batch_size}, dim={dim}, method={method}) ==="
    )

    # Initialize RHS with fixed seed for reproducibility
    key = jax.random.key(seed)
    rhs = LinearRHS(dim, rngs=nnx.Rngs(seed))

    # Generate random initial conditions
    x0_key, _ = jax.random.split(key)
    x0_shape = (batch_size,) + tuple(shape) if shape is not None else (batch_size, dim)
    x0_all = jax.random.normal(x0_key, (n_restart + 1,) + x0_shape)
    compile_x0 = x0_all[0]
    run_x0s = x0_all[1:]

    adaptive = (method == "rk45_adaptive")
    run_method = "rk45" if method == "rk45_adaptive" else method

    solve_fn = jax.jit(
        lambda x0: solve_ode_batched(
            rhs,
            x0,
            N_steps_max=N_steps,
            method=run_method,
            rtol=1e-4,
            adaptive=adaptive,
        )
    )

    ref_fn = jax.jit(jax.vmap(lambda _x: rhs.true_solution(1.,_x)))

    rhs.counter.reset()
    compile_start = time.perf_counter()
    compile_out = solve_fn(compile_x0)
    # ChatGPT addition: why is it here?
    _block_tree(compile_out)
    compile_time = time.perf_counter() - compile_start

    run_rows = []
    run_times = []
    rhs_counts = []
    rel_errors = []

    for idx, x0_batched in enumerate(run_x0s, start=1):
        rhs.counter.reset()
        run_start = time.perf_counter()
        x_batched = solve_fn(x0_batched)
        _block_tree(x_batched)
        run_time = time.perf_counter() - run_start
        rhs_calls = rhs.counter.get()

        x_ref = ref_fn(x0_batched)
        rel_error = float(batch_rel_err(x_batched, x_ref))
        passed = rel_error < tolerance

        run_times.append(run_time)
        rhs_counts.append(rhs_calls)
        rel_errors.append(rel_error)
        run_rows.append(
            {
                "restart": idx,
                "rel_error": rel_error,
                "runtime": run_time,
                "rhs_calls": rhs_calls,
                "passed": passed,
            }
        )

    max_error = max(rel_errors) if rel_errors else 0.0
    mean_error = float(jnp.mean(jnp.array(rel_errors))) if rel_errors else 0.0
    run_mean, run_std = _mean_std(run_times)
    rhs_mean, rhs_std = _mean_std(rhs_counts)
    success = all(row["passed"] for row in run_rows)

    print(f"Max rel error: {max_error:.2e}")
    print(f"Mean rel error: {mean_error:.2e}")
    print(f"Compile time: {compile_time:.4f}s")
    print(f"Runtime: {run_mean * 1e3:.3f}ms ± {run_std * 1e3:.3f}ms")
    print(f"RHS evaluations: {rhs_mean:.1f} ± {rhs_std:.1f}")

    if success:
        print("✓ ACCURACY TEST PASSED")
    else:
        print("✗ ACCURACY TEST FAILED")
        print(f"  Expected max rel error < {tolerance:.2e}, got {max_error:.2e}")
        print("  Debugging suggestions:")
        print("  - Try increasing N_steps_max for better accuracy")
        print("  - Check if rtol/atol are too loose")
        print("  - Verify the true_solution implementation")

    return (
        success,
        max_error,
        compile_time,
        run_mean,
        run_std,
        rhs_mean,
        rhs_std,
        run_rows,
    )


def test_autodiff_consistency(
    batch_size=3,
    dim=2,
    N_steps=30,
    seed=123,
    method="euler",
    jvp_tolerance=1e-2,
    grad_tolerance=1e-2,
    n_restart=10,
    shape=None,
):
    """Test b: autodiff consistency with reference solution."""
    print(
        f"\n=== Testing Autodiff Consistency (batch_size={batch_size}, dim={dim}, method={method}) ==="
    )

    # Initialize RHS with fixed seed
    key = jax.random.key(seed)
    rhs = LinearRHS(dim, rngs=nnx.Rngs(seed), with_counter=False)
    rhs_counter = LinearRHS(dim, rngs=nnx.Rngs(seed), with_counter=True)

    # Generate test input
    x0_key = jax.random.split(key)[1]
    x0_shape = (batch_size,) + tuple(shape) if shape is not None else (batch_size, dim)
    x0_all = jax.random.normal(x0_key, (n_restart + 1,) + x0_shape)
    compile_x0 = x0_all[0]
    run_x0s = x0_all[1:]

    graphdef, params, rest = nnx.split(rhs, nnx.Param, ...)
    param_tangents = jax.tree_util.tree_map(lambda value: jnp.sin(value), params)
    rest_tangent = jax.tree_util.tree_map(ad_util.zero_from_primal, rest)
    tang_rhs = nnx.merge(graphdef, param_tangents, rest_tangent)

    # Define batched function for testing
    adaptive = (method == "rk45_adaptive")
    run_method = "rk45" if method == "rk45_adaptive" else method
    def batched_func(rhs_module, x0):
        return solve_ode_batched(
            rhs_module,
            x0,
            N_steps_max=N_steps,
            method=run_method,
            adaptive=adaptive,
        )

    # Define reference function
    def reference_func(rhs_module, x0):
        return rhs_module.true_solution(1.0, x0) 
    
    reference_func = jax.vmap(reference_func, in_axes=(None, 0))

    def jvp_fn(module, x0):
        return jax.jvp(
            lambda m: batched_func(m, x0),
            (module,),
            (tang_rhs,),
        )[1]

    def grad_fn(module, x0):
        return nnx.grad(lambda m: jnp.mean(batched_func(m, x0) ** 2))(module)

    jitted_jvp = nnx.jit(jvp_fn)
    jitted_grad = nnx.jit(grad_fn)

    print("Testing forward mode (JVP)...")

    jvp_compile_start = time.perf_counter()
    jvp_compile_out = jitted_jvp(rhs, compile_x0)
    _block_tree(jvp_compile_out)
    jvp_compile_time = time.perf_counter() - jvp_compile_start

    print("\nTesting reverse mode (VJP/grad)...")

    grad_compile_start = time.perf_counter()
    grad_compile_out = jitted_grad(rhs, compile_x0)
    _block_tree(grad_compile_out)
    grad_compile_time = time.perf_counter() - grad_compile_start

    run_rows = []
    jvp_times = []
    grad_times = []
    rhs_counts = []
    jvp_errors = []
    grad_errors = []

    for idx, x0_batched in enumerate(run_x0s, start=1):
        # Compute the number of function evals for the solve;
        rhs_counter.counter.reset()
        adaptive = (method == "rk45_adaptive")
        run_method = "rk45" if method == "rk45_adaptive" else method
        rhs_forward = solve_ode_batched(
            rhs_counter,
            x0_batched,
            N_steps_max=N_steps,
            method=run_method,
            adaptive=adaptive,
        )
        _block_tree(rhs_forward)
        rhs_calls = rhs_counter.counter.get()

        jvp_start = time.perf_counter()
        jvp_batched = jitted_jvp(rhs, x0_batched)
        _block_tree(jvp_batched)
        jvp_time = time.perf_counter() - jvp_start

        jvp_ref = jax.jvp(
            lambda module: reference_func(module, x0_batched),
            (rhs,),
            (tang_rhs,),
        )[1]

        jvp_error = float(batch_rel_err(jvp_batched, jvp_ref))

        grad_start = time.perf_counter()
        grad_batched = jitted_grad(rhs, x0_batched)
        _block_tree(grad_batched)
        grad_time = time.perf_counter() - grad_start

        grad_ref = nnx.grad(
            lambda module: jnp.mean(reference_func(module, x0_batched) ** 2)
        )(rhs)

        _, grad_batched_params, _ = nnx.split(grad_batched, nnx.Param, ...)
        _, grad_ref_params, _ = nnx.split(grad_ref, nnx.Param, ...)
        grad_diffs = jax.tree_util.tree_map(
            lambda left, right: batch_rel_err(left, right),
            grad_batched_params,
            grad_ref_params,
        )
        grad_error = float(jnp.max(jnp.array(jax.tree_util.tree_leaves(grad_diffs))))

        passed = jvp_error < jvp_tolerance and grad_error < grad_tolerance
        run_rows.append(
            {
                "restart": idx,
                "jvp_error": jvp_error,
                "grad_error": grad_error,
                "jvp_time": jvp_time,
                "grad_time": grad_time,
                "rhs_calls": rhs_calls,
                "passed": passed,
            }
        )

        jvp_times.append(jvp_time)
        grad_times.append(grad_time)
        rhs_counts.append(rhs_calls)
        jvp_errors.append(jvp_error)
        grad_errors.append(grad_error)

    jvp_error = max(jvp_errors) if jvp_errors else 0.0
    grad_error = max(grad_errors) if grad_errors else 0.0
    jvp_mean, jvp_std = _mean_std(jvp_times)
    grad_mean, grad_std = _mean_std(grad_times)
    rhs_mean, rhs_std = _mean_std(rhs_counts)
    autodiff_success = all(row["passed"] for row in run_rows)

    print(f"JVP max error: {jvp_error:.2e}")
    print(f"JVP compile time: {jvp_compile_time:.4f}s")
    print(f"JVP runtime: {jvp_mean * 1e3:.3f}ms ± {jvp_std * 1e3:.3f}ms")
    print(f"Gradient max error: {grad_error:.2e}")
    print(f"Gradient compile time: {grad_compile_time:.4f}s")
    print(f"Gradient runtime: {grad_mean * 1e3:.3f}ms ± {grad_std * 1e3:.3f}ms")
    print(f"RHS evaluations: {rhs_mean:.1f} ± {rhs_std:.1f}")

    if autodiff_success:
        print("✓ AUTODIFF CONSISTENCY TEST PASSED")
    else:
        print("✗ AUTODIFF CONSISTENCY TEST FAILED")
        if jvp_error >= jvp_tolerance:
            print(f"  Expected JVP error < {jvp_tolerance:.2e}, got {jvp_error:.2e}")
        if grad_error >= grad_tolerance:
            print(f"  Expected grad error < {grad_tolerance:.2e}, got {grad_error:.2e}")

    return (
        autodiff_success,
        jvp_error,
        grad_error,
        jvp_compile_time,
        grad_compile_time,
        jvp_mean,
        jvp_std,
        grad_mean,
        grad_std,
        rhs_mean,
        rhs_std,
        run_rows,
    )


def run_comprehensive_tests(
    methods=None,
    tolerance=1e-2,
    steps=None,
    batch_size=None,
    dim=None,
    jvp_tolerance=None,
    grad_tolerance=None,
    n_restart=10,
    shapes=None,
):
    """Run tests with various configurations."""
    print("=" * 60)
    print("COMPREHENSIVE BATCHED SOLVER TESTS")
    print("=" * 60)

    base_config = {"batch_size": 8, "dim": 2, "N_steps": 25, "seed": 789}
    methods = methods or ["rk45", "rk45_adaptive", "euler", "heun"]
    jvp_tolerance = tolerance if jvp_tolerance is None else jvp_tolerance
    grad_tolerance = tolerance if grad_tolerance is None else grad_tolerance

    if batch_size is not None:
        base_config["batch_size"] = batch_size
    if dim is not None:
        base_config["dim"] = dim

    if shapes:
        test_configs = []
        for shape in shapes:
            shape_dim = int(jnp.prod(jnp.array(shape)))
            config = {**base_config, "dim": shape_dim, "shape": shape}
            test_configs.append(config)
    else:
        test_configs = [base_config]

    report_lines = [
        "# Batched ODE Test Report",
        "",
    ]
    all_passed = True

    if steps is not None and len(steps) != len(methods):
        raise ValueError("--steps must match number of methods")

    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*20} Test Suite {i} {'='*20}")
        report_lines.append(f"## Test Suite {i}")
        report_lines.append(f"Config: `{config}`")
        if config.get("shape") is not None:
            report_lines.append(f"Shape: `{config['shape']}`")
        report_lines.append("")

        suite_passed = True
        step_values = steps or [config["N_steps"]] * len(methods)

        for method, step_value in zip(methods, step_values):
            print(f"\n--- Method: {method} (steps={step_value}) ---")
            report_lines.append(f"### Method: `{method}` (steps={step_value})")
            if config.get("shape") is not None:
                report_lines.append(f"Shape: `{config['shape']}`")
            report_lines.append("")

            acc_success = False
            auto_success = False
            acc_error = None
            acc_compile_time = None
            acc_run_mean = None
            acc_run_std = None
            acc_rhs_mean = None
            acc_rhs_std = None
            acc_rows = None
            jvp_err = None
            grad_err = None
            jvp_compile_time = None
            grad_compile_time = None
            jvp_run_mean = None
            jvp_run_std = None
            grad_run_mean = None
            grad_run_std = None
            auto_rhs_mean = None
            auto_rhs_std = None
            auto_rows = None

            try:
                (
                    acc_success,
                    acc_error,
                    acc_compile_time,
                    acc_run_mean,
                    acc_run_std,
                    acc_rhs_mean,
                    acc_rhs_std,
                    acc_rows,
                ) = test_batched_solver_accuracy(
                    **{**config, "N_steps": step_value},
                    method=method,
                    tolerance=tolerance,
                    n_restart=n_restart,
                )
            except Exception:
                all_passed = False
                suite_passed = False
                acc_success = False
                report_lines.append("**Accuracy:** EXCEPTION")
                report_lines.append("<details>")
                report_lines.append("<summary>Traceback</summary>")
                report_lines.append("")
                report_lines.append("```text")
                report_lines.append(traceback.format_exc())
                report_lines.append("```")
                report_lines.append("</details>")
                report_lines.append("")

            try:
                (
                    auto_success,
                    jvp_err,
                    grad_err,
                    jvp_compile_time,
                    grad_compile_time,
                    jvp_run_mean,
                    jvp_run_std,
                    grad_run_mean,
                    grad_run_std,
                    auto_rhs_mean,
                    auto_rhs_std,
                    auto_rows,
                ) = test_autodiff_consistency(
                    **{**config, "N_steps": step_value},
                    method=method,
                    jvp_tolerance=jvp_tolerance,
                    grad_tolerance=grad_tolerance,
                    n_restart=n_restart,
                )
            except Exception:
                all_passed = False
                suite_passed = False
                auto_success = False
                report_lines.append("**Autodiff:** EXCEPTION")
                report_lines.append("<details>")
                report_lines.append("<summary>Traceback</summary>")
                report_lines.append("")
                report_lines.append("```text")
                report_lines.append(traceback.format_exc())
                report_lines.append("```")
                report_lines.append("</details>")
                report_lines.append("")

            suite_passed = suite_passed and acc_success and auto_success

            if acc_error is not None:
                print(
                    "Accuracy: "
                    f"{'PASS' if acc_success else 'FAIL'} (error: {acc_error:.2e}, "
                    f"compile: {acc_compile_time:.4f}s, run: {acc_run_mean * 1e3:.3f}ms ± {acc_run_std * 1e3:.3f}ms, "
                    f"rhs: {acc_rhs_mean:.1f} ± {acc_rhs_std:.1f})"
                )
                report_lines.append(
                    "**Accuracy:** "
                    f"{'PASS' if acc_success else 'FAIL'} (error: {acc_error:.2e}, "
                    f"compile: {acc_compile_time:.4f}s, run: {acc_run_mean * 1e3:.3f}ms ± {acc_run_std * 1e3:.3f}ms, "
                    f"rhs: {acc_rhs_mean:.1f} ± {acc_rhs_std:.1f})"
                )
                if acc_rows:
                    report_lines.append("<details>")
                    report_lines.append("")
                    report_lines.append("<summary>Accuracy restarts</summary>")
                    report_lines.append("")
                    report_lines.append("| Restart | Max Rel Error | Mean Rel Error | Runtime (ms) | RHS Calls | Pass |")
                    report_lines.append("| --- | --- | --- | --- | --- | --- |")
                    for row in acc_rows:
                        report_lines.append(
                            f"| {row['restart']} | {row['rel_error']:.2e} | {row['rel_error']:.2e} | "
                            f"{row['runtime'] * 1e3:.3f} | {row['rhs_calls']} | {row['passed']} |"
                        )
                    report_lines.append("")
                    report_lines.append("</details>")
                    report_lines.append("")
            if jvp_err is not None and grad_err is not None:
                total_compile = None
                if jvp_compile_time is not None and grad_compile_time is not None:
                    total_compile = jvp_compile_time + grad_compile_time
                compile_label = f"{total_compile:.4f}s" if total_compile is not None else "n/a"
                print(
                    "Autodiff: "
                    f"{'PASS' if auto_success else 'FAIL'} (JVP: {jvp_err:.2e}, Grad: {grad_err:.2e}, "
                    f"compile: {compile_label}, "
                    f"run: {jvp_run_mean * 1e3:.3f}ms ± {jvp_run_std * 1e3:.3f}ms / "
                    f"{grad_run_mean * 1e3:.3f}ms ± {grad_run_std * 1e3:.3f}ms, "
                    f"rhs: {auto_rhs_mean:.1f} ± {auto_rhs_std:.1f})"
                )
                report_lines.append(
                    "**Autodiff:** "
                    f"{'PASS' if auto_success else 'FAIL'} (JVP: {jvp_err:.2e}, Grad: {grad_err:.2e}, "
                    f"compile: {compile_label}, "
                    f"run: {jvp_run_mean * 1e3:.3f}ms ± {jvp_run_std * 1e3:.3f}ms / "
                    f"{grad_run_mean * 1e3:.3f}ms ± {grad_run_std * 1e3:.3f}ms, "
                    f"rhs: {auto_rhs_mean:.1f} ± {auto_rhs_std:.1f})"
                )
                if auto_rows:
                    report_lines.append("<details>")
                    report_lines.append("")
                    report_lines.append("<summary>Autodiff restarts</summary>")
                    report_lines.append("")
                    report_lines.append(
                        "| Restart | JVP Error | Grad Error | JVP Time (ms) | Grad Time (ms) | RHS Calls | Pass |"
                    )
                    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                    for row in auto_rows:
                        report_lines.append(
                            f"| {row['restart']} | {row['jvp_error']:.2e} | {row['grad_error']:.2e} | "
                            f"{row['jvp_time'] * 1e3:.3f} | {row['grad_time'] * 1e3:.3f} | {row['rhs_calls']} | {row['passed']} |"
                        )
                    report_lines.append("")
                    report_lines.append("</details>")
                    report_lines.append("")

            report_lines.append("")

        all_passed = all_passed and suite_passed
        print(f"\n--- Test Suite {i} Summary ---")
        print(f"Overall: {'PASS' if suite_passed else 'FAIL'}")
        report_lines.append(f"**Overall:** {'PASS' if suite_passed else 'FAIL'}")
        report_lines.append("")

    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nGeneral debugging suggestions:")
        print("- Check JAX installation: try `pip install --upgrade jax jaxlib`")
        print("- Enable X64: set `JAX_ENABLE_X64=1` environment variable")
        print("- Check Flax version compatibility with JAX")
        print("- Verify the ODE solver implementation matches reference")
    print(f"{'='*60}")

    with open("BATCHED_ODE_TEST_RESULTS.md", "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batched ODE tests.")
    parser.add_argument(
        "--methods",
        default="rk45,rk45_adaptive,euler,heun",
        help="Comma-separated list of methods to test.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Accuracy tolerance for solver tests.",
    )
    parser.add_argument(
        "--jvp-tolerance",
        type=float,
        default=None,
        help="JVP tolerance (defaults to --tolerance).",
    )
    parser.add_argument(
        "--grad-tolerance",
        type=float,
        default=None,
        help="Gradient tolerance (defaults to --tolerance).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of N_steps_max aligned with methods.",
    )
    parser.add_argument(
        "--n-restart",
        type=int,
        default=10,
        help="Number of restarts for runtime measurements.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for all suites.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Override dimension for all suites.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Semicolon-separated shapes with comma-separated dims (e.g. '4,5;6,7,8').",
    )

    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    allowed_methods = {"rk45", "euler", "heun", "rk45_adaptive"}
    unknown = [method for method in methods if method not in allowed_methods]
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. Allowed: {sorted(allowed_methods)}"
        )

    steps = None
    if args.steps is not None:
        steps = [int(value) for value in args.steps.split(",") if value.strip()]

    shapes = None
    if args.shapes is not None:
        shapes = []
        for raw_shape in args.shapes.split(";"):
            dims = [int(value) for value in raw_shape.split(",") if value.strip()]
            if not dims:
                continue
            shapes.append(tuple(dims))

    if args.dim is not None and shapes is not None:
        raise ValueError("--dim and --shapes are mutually exclusive")

    success = run_comprehensive_tests(
        methods=methods,
        tolerance=args.tolerance,
        steps=steps,
        batch_size=args.batch_size,
        dim=args.dim,
        jvp_tolerance=args.jvp_tolerance,
        grad_tolerance=args.grad_tolerance,
        n_restart=args.n_restart,
        shapes=shapes,
    )
    exit(0 if success else 1)
