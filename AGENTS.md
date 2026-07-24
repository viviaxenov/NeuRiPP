# Repository Instructions

## Setup
- This is a setuptools `src/` layout package: install for development with `pip install -e .`; optional conda deps are in `requirements_conda.txt`.
- The import package is lowercase `neuripp` even though the project name is `NeuRiPP`.
- Runtime deps are only declared through `requirements.txt`; there is no lockfile or configured lint/typecheck tool.

## Verification
- Do not run broad `pytest` casually: several `tests/test_*.py` files execute JAX training/plotting work at import time rather than behaving like lightweight pytest tests.
- ODE verification is script-oriented: `python tests/test_ode.py`. It writes `BATCHED_ODE_TEST_RESULTS.md` in the repo root.
- For a quick focused ODE smoke test, use `python tests/test_ode.py --methods euler --steps 4 --n-restart 1 --batch-size 1 --dim 1 --tolerance 1e1 --jvp-tolerance 1e1 --grad-tolerance 1e1`.

## Architecture
- Core ODE code lives in `src/neuripp/_ode/_ode.py`; supported solver names are `rk45`, `euler`, and `heun`, with adaptive RK45 enabled by passing `adaptive=True`.
- `ParametricPushforward` in `src/neuripp/parametric_pushforward/parametric_pushforward.py` expects a Flax `nnx.Module` RHS with a `dim` attribute and call signature like `rhs(t, x, *args)` for batched `x`.
- Package `__init__.py` files are empty; import concrete modules directly instead of assuming public re-exports.
- Optimization methods under `src/neuripp/methods/` use Flax NNX parameter splitting/merging; preserve pytree compatibility when changing model state.

## Docs
- Docs are Sphinx/NBSphinx. Install docs extras with `pip install -e '.[docs]'`, then build from `docs/` with `make html`; output goes to `docs/build/html`.
- If adding modules that need API docs, regenerate stubs with `sphinx-apidoc -o docs/source/ src/neuripp/`.
- Notebook examples are sourced from `examples/notebooks`; add new entries manually to `docs/source/_include/examples.rst`.

## Generated Files
- Avoid committing local JAX caches, generated PDFs, GIFs, or benchmark reports unless explicitly requested; current tests can create these under `tests/` or the repo root.
