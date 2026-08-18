# Benchmark Runner Specification

## 1. Purpose

Define a benchmark runner.

The runner should:

- Load a JSON configuration describing one benchmark problem, architecture grids, method grids, shared parameters, plotting options, and optional parallel execution settings.
- Expand the configuration into a deterministic list of planned runs.
- Execute planned runs using the configured architecture, method, and problem definitions.
- Persist per-run results, metadata, status files, error details, arrays, and checkpoint artifacts into structured per-run output directories.
- Generate aggregate plots from completed runs.
- Support a plot-only mode that regenerates plots from saved results without rerunning benchmarks.
- Support sequential execution and multi-process execution with one active run per worker.

The runner should separate orchestration from benchmark-specific logic. Model construction, objective construction, method execution, checkpoint format, and plot content should be implemented behind clearly defined project functions.

## 2. CLI Contract

The benchmark runner should expose a command-line interface with these options:

```bash
python examples/benchmark_runner.py \
  --config path/to/benchmark_config.json \
  [--plot-only] \
  [--run-id 3] \
  [--run-name experiment_name]
```

Required behavior:

- `--config` points to the JSON benchmark configuration.
- If omitted, it may default to a conventional config path such as `benchmark_config.json`.
- `--plot-only` skips benchmark execution and regenerates plots from existing saved results.
- `--run-id` runs exactly one planned run by numeric `run_index`.
- `--run-name` uses a stable sanitized output directory name instead of creating a timestamped session directory.
- `--plot-only` and `--run-id` must be mutually exclusive.

Output path behavior:

- Without `--run-name`, create a new timestamped session directory under `output_root`.
- With `--run-name`, write into `output_root/<sanitized_run_name>/`.
- With `--run-id`, write single-run results using the selected run's output layout.
- Without `--run-id`, write the full benchmark into the selected session directory.

The CLI should fail early with clear errors for invalid flag combinations, missing config files, invalid JSON, or missing saved results in plot-only mode. On successful completion, the CLI should print a summary and exit with code 0. On error, it should print an error message and exit with code 1.

## 3. JSON Config Contract

The runner should load a single JSON config file.

Required top-level keys:

```json
{
  "output_root": "results/benchmarks",
  "common_params": {},
  "problem": {},
  "architectures": [],
  "methods": [],
  "plotting": {}
}
```

Optional top-level keys:

```json
{
  "parallel": {},
  "config": []
}
```

Recommended top-level ordering in examples:

```json
{
  "output_root": "results/benchmarks",
  "common_params": {},
  "problem": {},
  "architectures": [],
  "methods": [],
  "plotting": {},
  "parallel": {}
}
```

Validation rules:

- `output_root` must be a string path.
- `common_params` must be an object.
- `problem` must be an object.
- `architectures` must be a non-empty list.
- `methods` must be a non-empty list.
- `plotting` must be an object.
- `parallel`, if present, must be an object.
- `config`, if present, must be a list of `[flag, value]` pairs. Each flag must be a string. Each value is passed through to `jax.config.update(flag, value)` unchanged.
- Do not support multiple problems in one config.
- Do not support deprecated aliases such as `problems` or `distributions`.
- Unknown top-level keys may either be rejected or preserved as metadata, but the implementation should choose one behavior and document it.
- Config validation should happen before any benchmark run starts.

General config semantics:

- `common_params` contains shared values appended into every planned run's `method_kwargs`.
- `problem` contains the single benchmark problem definition used for every planned run.
- `architectures` contains NeuralODE right-hand-side architecture templates.
- `methods` contains method templates and method hyperparameter grids.
- `plotting` contains plot generation settings and style rules.
- `parallel` controls worker/process execution behavior.
- `config` contains JAX configuration flag/value pairs applied in every worker after `import jax`, before device discovery or runtime imports. Example: `["jax_compilation_cache_dir", "/tmp/jax_cache"]`.

Recommended normalized internal representation:

```python
{
    "output_root": Path(...),
    "common_params": dict,
    "problem": dict,
    "architectures": list[dict],
    "methods": list[dict],
    "plotting": dict,
    "parallel": {
        "n_workers": 1,
    },
}
```

The implementation should apply defaults during config loading, not during execution, so later stages can assume a validated normalized config.

## 4. Problem Schema

Each config defines exactly one benchmark problem using a `functional` plus a `distribution`.

```json
{
  "problem": {
    "functional": {
      "kind": "KL"
    },
    "distribution": {
      "name": "gaussian",
      "mean_value": 2.0,
      "sigma_diag": [1000.0, 10.0]
    }
  }
}
```

Required fields:

- `problem.functional`
- `problem.functional.kind`
- `problem.distribution`

Supported functional kinds, exactly as written:

- `KL`
- `MMD`
- `CrossEntropy`

Validation rules:

- `problem` must be an object.
- `problem.functional` must be an object.
- `problem.functional.kind` must be exactly one of `KL`, `MMD`, or `CrossEntropy`.
- Lowercase names, hyphenated names, and aliases must fail validation.
- `problem.distribution` must be either a string or an object.
- If `problem.distribution` is a string, normalize it to `{ "name": "<string>" }`.
- If `problem.distribution` is an object, preserve its fields.
- Do not support `functional.name`; require `functional.kind`.
- Unsupported distribution names should fail during problem construction.

Functional-specific fields for `KL`:

```json
{
  "kind": "KL"
}
```

Functional-specific fields for `MMD`:

```json
{
  "kind": "MMD",
  "bw_multipliers": [0.5, 1.0, 2.0],
  "bandwidth_samples": 2000
}
```

Functional-specific fields for `CrossEntropy`:

```json
{
  "kind": "CrossEntropy"
}
```

Distribution examples:

```json
{
  "name": "gaussian",
  "mean_value": 2.0,
  "sigma_diag": [1000.0, 10.0]
}
```

```json
{
  "name": "two_spirals",
  "n_samples": 1000,
  "resample_each": 1,
  "seed": 3
}
```

```json
{
  "name": "eight_gaussians",
  "n_samples": 1000,
  "resample_each": 1,
  "seed": 3
}
```

The runner should pass the validated `problem` object to problem-construction logic, which returns the concrete objective/functional object and metadata to persist.

## 5. Methods And Architectures Grid Schema

The config defines one benchmark problem, one or more architecture templates, and one or more method templates.

Example:

```json
{
  "output_root": "results/benchmarks",
  "common_params": {
    "max_iterations": 300,
    "batch_size": 100,
    "master_seed": 0
  },
  "problem": {
    "functional": {
      "kind": "KL"
    },
    "distribution": {
      "name": "gaussian",
      "mean_value": 2.0,
      "sigma_diag": [1000.0, 10.0]
    }
  },
  "architectures": [
    {
      "rhs": {
        "model": "MLP",
        "n_hidden": [2, 3],
        "dim_hidden": [16, 32],
        "activation": "tanh"
      },
      "N_monte_carlo": 128,
      "divergence_method": "hutchinson",
      "ode_method": "rk45",
      "ode_nstep_max": 12,
      "h_max": 0.3
    }
  ],
  "methods": [
    {
      "method": "anderson",
      "n_restarts": 4,
      "relaxation": [1.25, 1.8],
      "regularization_factor": [1e-5, 1e-3],
      "regularization_method": "l2",
      "natural_grad_clipping_threshold": [null, 1.0],
      "step_size": 0.05
    },
    {
      "method": "ngd",
      "n_restarts": 4,
      "step_size": [0.05, 0.1],
      "natural_grad_clipping_threshold": [null, 1.0],
      "linear_solver_regularization": 0.001
    },
    {
      "method": "adamw",
      "n_restarts": 4,
      "learning_rate": [0.001, 0.0003],
      "weight_decay": 0.0001
    }
  ],
  "plotting": {},
  "parallel": {
    "gpu_ids": [0],
    "max_parallel": {
      "anderson": 8,
      "adamw": 32
    }
  }
}
```

Architecture validation rules:

- `architectures` must be a non-empty list.
- Each architecture item must be an object.
- Each architecture item must contain an `rhs` object.
- `architecture.rhs.model` must be a string matching an RHS class name from `tests/test_rhs.py`.
- The runner should not validate architecture-specific fields beyond basic grid validity and the required `rhs` structure.
- Empty list-valued architecture arguments must fail validation.
- Architecture-specific validity belongs to model-construction logic.
- Values inside `architecture.rhs` are passed to the RHS constructor together with the inferred dimension.
- Direct architecture keys: `N_monte_carlo`, `divergence_method`, `ode_method`, `ode_nstep_max`. These are passed to `ParametricPushforward`.
- Any remaining top-level architecture keys (e.g. `h_max`, `N_iter_to_accept`, `adaptive`) are collected into `ode_kwargs` and passed to `ParametricPushforward`.

Architecture grid expansion:

- For each architecture object, inspect every top-level field and every field inside `rhs`.
- If a value is a list, treat it as a grid axis.
- If a value is not a list, treat it as a scalar shared by all expanded architectures.
- Compute the Cartesian product over all list-valued fields, including list-valued `rhs` fields.
- Preserve architecture template order from the `architectures` list.
- Preserve field order from each architecture object.
- Preserve field order from each nested `rhs` object.
- A template with no list values produces one expanded architecture.

Method schema:

```json
{
  "methods": [
    {
      "method": "anderson",
      "relaxation": [1.25, 1.8],
      "regularization": [1e-5, 1e-3]
    }
  ]
}
```

Method validation rules:

- `methods` must be a non-empty list.
- Each method item must be an object.
- Each method item must contain `method`.
- `method` must be a string.
- The runner should validate only that `method` exists in `str_to_method`.
- Do not support method aliases.
- Do not validate method-specific keyword arguments.
- Empty list-valued method arguments must fail validation.

Method dispatch contract:

- The project must define `str_to_method`.
- `str_to_method` maps method names to method initialization or execution functions.
- The `method` field selects the callable: `method_fn = str_to_method[method_name]`.
- All non-`method` fields become `expanded_method_kwargs` after grid expansion.
- Method kwargs accepted by `get_<METHOD>(...)` are treated as factory kwargs for execution grouping and analysis method grouping.
- JSON `null` values for method kwargs, such as `natural_grad_clipping_threshold`, should be preserved as Python `None`.

Method grid expansion:

- For each method object, inspect every field except `method`.
- If a value is a list, treat it as a grid axis.
- If a value is not a list, treat it as a scalar shared by all expanded runs.
- Compute the Cartesian product over all list-valued fields.
- Preserve method template order from the `methods` list.
- Preserve argument order from each method object.
- Generate expanded methods in deterministic product order.
- A method object with no extra fields produces one expanded method with empty kwargs.

Combined run expansion:

- First expand all architecture templates into expanded architecture configs.
- Then expand all method templates into expanded method configs.
- Then take the outer Cartesian product of expanded architectures, expanded methods, and expanded problems.
- Each architecture is trained with each expanded method for each expanded problem.
- Preserve deterministic order: architecture template order, architecture grid product order, method template order, method grid product order.

Common parameter handling:

- After selecting an expanded architecture and expanded method, append `common_params` into `method_kwargs`.
- Method-specific kwargs override `common_params`.

```python
method_kwargs = {
    **common_params,
    **expanded_method_kwargs,
}
```

Architecture should be passed separately from method kwargs.

Planned run record:

```python
{
    "run_index": 0,
    "run_id": "run_0000",
    "restart_index": 0,
    "problem": problem,
    "architecture": expanded_architecture,
    "method": "anderson",
    "method_kwargs": {
        "max_iterations": 300,
        "batch_size": 100,
        "master_seed": 0,
        "relaxation": 1.25,
        "regularization_factor": 1e-5,
        "regularization_method": "l2",
        "natural_grad_clipping_threshold": null,
        "step_size": 0.05,
    },
}
```

## 6. Run Planning

The runner should convert the validated config into a deterministic list of planned runs before execution starts.

Inputs to planning:

- `problem`: the single validated problem object.
- `architectures`: expanded from architecture templates.
- `methods`: expanded from method templates.
- `common_params`: appended into each run's `method_kwargs`.

Restart policy:

- Repeated runs are represented with `methods[].n_restarts`.
- Each planned run gets a deterministic `restart_index` in `0..n_restarts-1`.
- `common_params.master_seed` is the single benchmark-level seed used to initialize vectorized chunks consistently.
- `problem.distribution.seed` is a scalar distribution parameter, not a benchmark expansion axis.
- Aggregate plotting collapses runs that differ only by `restart_index`.
- Runs that differ in any factory kwarg accepted by `get_<METHOD>(...)` are treated as different methods for execution grouping and analysis.

Expansion order:

1. Expand each architecture template.
2. Expand each method template.
3. Merge `common_params` into each expanded method kwargs, with method-specific kwargs taking precedence.
4. Take the outer Cartesian product of expanded architectures, expanded methods, and expanded problems.

Ordering must be deterministic:

1. Preserve architecture template order from `architectures`.
2. Preserve architecture parameter order within each template.
3. Preserve architecture Cartesian product order.
4. Preserve method template order from `methods`.
5. Preserve method parameter order within each template.
6. Preserve method Cartesian product order.

Each planned run should contain:

```python
{
    "run_index": 0,
    "run_id": "run_0000",
    "problem": problem,
    "architecture": expanded_architecture,
    "method": "anderson",
    "method_kwargs": {
        **common_params,
        **expanded_method_kwargs,
    },
}
```

Run identity:

- `run_index` is a zero-based integer assigned after full expansion.
- `run_id` should be stable and path-safe.
- Recommended minimal format: `run_0000`.

Single-run selection:

- `--run-id N` selects the planned run whose `run_index == N`.
- If `N` is outside the valid range, fail before execution with a clear error.
- In single-run mode, only that planned run is executed.
- The selected run should keep its original `run_index` and `run_id` from the full plan.

Architecture and method labels:

- The runner may compute human-readable labels for logs and plots.
- Labels should be derived from the expanded architecture and method kwargs.
- Labels should not affect run identity unless explicitly designed to be stable.

Planning should be pure:

- It should not build models.
- It should not initialize methods.
- It should not create output directories.
- It should not write files.
- It should only transform validated config into planned run dictionaries.

## 7. Parallel Execution Contract

The runner should create a master process plus `parallel.n_workers` worker processes.

```json
{
  "parallel": {
    "n_workers": 4
  }
}
```

Validation rules:

- `parallel` is optional.
- If omitted, use `{ "n_workers": 1 }`.
- `parallel.n_workers` must be an integer greater than or equal to `1`.
- `n_workers == 1` may still use the same worker-loop logic in the main process, or may use a simplified sequential path. The behavior should be documented.

Execution model:

- The master process owns the full planned-run stack.
- Each planned run is a fully expanded dictionary containing all parameters needed for one run.
- The master starts `parallel.n_workers` workers.
- Workers repeatedly pull one planned run from the shared task queue.
- A worker executes exactly one run at a time.
- When the queue is empty, the worker emits an exit message and exits.
- The master waits for all workers to exit, then finalizes the benchmark.

Recommended task shape:

```python
{
    "run_index": 0,
    "run_id": "run_0000",
    "problem": {...},
    "architecture": {...},
    "method": "anderson",
    "method_kwargs": {...},
    "output_dir": "...",
}
```

Worker cycle:

```python
while True:
    planned_run = try_get_task()
    if no_task_available:
        send_message({
            "event": "worker_empty_queue",
            "worker_id": worker_id,
        })
        break

    send_message({
        "event": "run_started",
        "worker_id": worker_id,
        "run_id": planned_run["run_id"],
    })

    write_expanded_run_config(planned_run)

    try:
        write_intermediate_status(planned_run, status="running")
        result = execute_run(planned_run)
        write_run_outputs(planned_run, result)
        write_intermediate_status(planned_run, status="success")
        send_message({
            "event": "run_success",
            "worker_id": worker_id,
            "run_id": planned_run["run_id"],
        })
    except Exception as exc:
        write_error_file(planned_run, exc, traceback.format_exc())
        write_intermediate_status(planned_run, status="failed", error=str(exc))
        send_message({
            "event": "run_error",
            "worker_id": worker_id,
            "run_id": planned_run["run_id"],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
```

Per-run expanded config:

- Each worker must write the fully expanded config for its run.
- This file should be written before execution starts.
- It should contain the exact planned-run dictionary after all grid expansion and common-parameter merging.
- Recommended filename: `<run_output_dir>/expanded_config.json`.

Intermediate status file:

- Each worker should write an intermediate status file for the run.
- Recommended filename: `<run_output_dir>/status.json`.
- Recommended states: `running`, `success`, `failed`.

Output write rules:

- Workers may write output files directly, but only inside unique per-run output directories.
- Workers must not write concurrently to a shared aggregate file.
- If aggregate results are needed, the master should build them after all workers finish by reading per-run outputs.
- Per-run output directories must be unique and deterministic, for example `runs/run_0000/`.

Recommended worker messages:

```python
{"event": "run_started", "worker_id": 0, "run_id": "run_0000"}
{"event": "run_success", "worker_id": 0, "run_id": "run_0000"}
{"event": "run_error", "worker_id": 0, "run_id": "run_0000", "error": "...", "traceback": "..."}
{"event": "worker_empty_queue", "worker_id": 0}
{"event": "worker_exit", "worker_id": 0}
```

Master process responsibilities:

- Build and validate the planned-run stack.
- Create the task queue and message queue.
- Create deterministic per-run output directories.
- Start `parallel.n_workers` workers.
- Monitor worker messages.
- Track success/failure counts.
- Detect unexpected worker death.
- Wait for all workers to exit.
- Optionally assemble aggregate summaries/plots from per-run output files.

Worker process responsibilities:

- Pull planned runs from the queue until empty.
- Write per-run `expanded_config.json`.
- Write/update per-run `status.json`.
- Execute the run.
- Write per-run outputs on success.
- Send structured messages to the master.
- Exit cleanly when no tasks remain.

### JAX Compilation Cache

The runner should use long-lived worker processes. Each worker should process multiple runs from the queue so process-local JAX compilation caches can be reused.

The runner should not assume in-memory JAX compilation caches are shared across worker processes.

TODO: Configure JAX persistent compilation cache for this project so worker processes can reuse compatible compiled artifacts from disk. Document the required setup, environment variables, cache directory, and any limitations.

Run planning should preserve enough metadata to optionally group shape-compatible runs on the same worker.

## 8. Failure Handling

The runner should treat failures as per-run outcomes, not as reasons to stop the full benchmark.

General behavior:

- If a run fails, the responsible worker records the failure and exits that run cleanly.
- Other workers continue processing available runs.
- The benchmark continues until the task queue is empty and all workers have exited.
- The master process reports final success/failure counts.

Required failed-run artifacts:

```text
<run_output_dir>/
  expanded_config.json
  status.json
  error.txt
```

`expanded_config.json` should be written before execution starts.

`status.json` should be written with `status: running` before execution starts.

On failure, the worker must write the error message and traceback to disk inside the failed run's output directory.

Failed `status.json` example:

```json
{
  "status": "failed",
  "run_id": "run_0000",
  "worker_id": 0,
  "error": "Exception message",
  "traceback_path": "error.txt"
}
```

`error.txt` example:

```text
Run failed: run_0000
Worker: 0

Error:
Exception message

Traceback:
<full Python traceback>
```

Successful `status.json` example:

```json
{
  "status": "success",
  "run_id": "run_0000",
  "worker_id": 0
}
```

Running `status.json` example:

```json
{
  "status": "running",
  "run_id": "run_0000",
  "worker_id": 0,
  "started_at": "..."
}
```

Worker message on failure:

```python
{
    "event": "run_error",
    "worker_id": 0,
    "run_id": "run_0000",
    "error": "...",
    "traceback": "...",
}
```

Unexpected worker death:

- The master should detect if a worker process exits unexpectedly.
- If the worker had an active run, that run should be marked failed by the master if the worker did not already write a final status.
- The master should write or update that run's `status.json` if possible.
- If the worker crashes before writing `error.txt`, the master should write a best-effort `error.txt` using the information it has, such as worker ID, active run ID, exit code, and detection time.
- The master should continue monitoring other workers.
- If the task queue still contains runs but all workers have died, the master should report a clear orchestration failure.

Queue exhaustion:

- If a worker cannot pick a task because the queue is empty, it should emit `{ "event": "worker_empty_queue", "worker_id": 0 }`.
- Then it should exit normally.
- Empty queue is not a failure.

Final summary:

```json
{
  "success": 10,
  "failed": 2,
  "total": 12
}
```

No fail-fast behavior should be included in this specification.

## 9. Checkpoint Contract

Each successful run must write two model checkpoints:

```text
<session_dir>/
  runs/
    run_0000/
      checkpoints/
        best/
        last/
```

Required checkpoints:

- `checkpoints/last/`: model state at the end of training.
- `checkpoints/best/`: model state with the best selected metric value during training.

Best-checkpoint criterion:

- The implementation must define and document the criterion used for `best`.
- Recommended default: minimum loss.
- The criterion should be recorded in run metadata or `status.json`.

Checkpoint write behavior:

- `last` should be written at the end of a successful run.
- `best` should be written whenever a new best model is observed, or once at the end from the saved best state.
- Checkpoints should be written only inside the unique per-run output directory.
- Workers must not write checkpoints into shared paths.
- If training succeeds but checkpoint writing fails, mark the run as failed unless result-only runs are explicitly supported.

Checkpoint format:

- The checkpoint serialization format uses Orbax `StandardCheckpointer` with full Flax NNX state.
- The format must support reconstructing the model from `expanded_config.json` plus checkpoint contents.
- A `metadata.json` file is written alongside each checkpoint directory recording the run_id, checkpoint key, format, criterion, and metric value.

## 10. Loading API

The runner module should expose importable functions for loading completed benchmark outputs from other code.

### Load Completed Runs

```python
def load_completed_runs(session_dir: str | Path) -> list[dict]:
    ...
```

Required behavior:

- Scan `<session_dir>/runs/*/`.
- Read each run's `status.json`.
- Include only runs with `status: "success"`.
- Read each included run's `expanded_config.json`.
- Read the run's saved arrays, for example from `arrays.npz`.
- Return a list of dictionaries.

Recommended return shape:

```python
[
    {
        "run_id": "run_0000",
        "run_dir": Path(".../runs/run_0000"),
        "expanded_config": {...},
        "arrays": {
            "loss": np.ndarray,
            "grad_norm": np.ndarray,
        },
    }
]
```

Notes:

- The exact array names are project-specific.
- The function should preserve all arrays written by the run.
- Failed and incomplete runs should be skipped by default.
- If no successful runs are found, return an empty list or raise a clear error; the implementation should choose and document one behavior.

### Load Checkpointed Model

```python
def load_model_checkpoint(
    session_dir: str | Path,
    run_id: str,
    key: Literal["last", "best"] = "last",
):
    ...
```

Required behavior:

- Validate `key` is either `"last"` or `"best"`.
- Locate `<session_dir>/runs/<run_id>/expanded_config.json`.
- Locate `<session_dir>/runs/<run_id>/checkpoints/<key>/`.
- Reconstruct the model architecture from `expanded_config.json`.
- Restore model parameters from the selected checkpoint.
- Return the restored model.
- If `run_id` is unknown, raise a clear error.
- If the selected checkpoint is missing, raise `FileNotFoundError`.
- If model reconstruction fails, raise a clear error.

## 11. Plot-Only Mode

Plot-only mode should use the Loading API rather than duplicating loading logic.

CLI:

```bash
python examples/benchmark_runner.py --config benchmark_config.json --plot-only
```

Behavior:

- Do not execute planned runs.
- Locate an existing benchmark session directory.
- Load successful runs via `runs = load_completed_runs(session_dir)`.
- Regenerate aggregate plots and summaries.
- If plots need restored models, use `model = load_model_checkpoint(session_dir, run_id, key="last")`.
- Do not modify per-run training results.
- Plot-only mode may overwrite plot files under the plot output directory.

Recommended input selection:

- If `--run-name` is provided, use `<output_root>/<sanitized_run_name>/`.
- Prefer also supporting `--output-dir path/to/session`.
- If neither is provided, locate the latest benchmark session under `output_root`.

Recommended per-run layout assumed by plot-only mode:

```text
<session_dir>/
  runs/
    run_0000/
      expanded_config.json
      status.json
      arrays.npz
      checkpoints/
        best/
        last/
    run_0001/
      expanded_config.json
      status.json
      arrays.npz
      checkpoints/
        best/
        last/
  plots/
```

Plot-only mode should:

- Scan `runs/*/status.json` through `load_completed_runs`.
- Select runs with `status: "success"`.
- Load their saved outputs.
- Rebuild aggregate plots under `<session_dir>/plots/`.

## 12. Implementation Checklist

Implementation steps:

- Implement CLI parsing for `--config`, `--plot-only`, `--run-id`, `--run-name`, and optionally `--output-dir`.
- Implement JSON config loading and validation.
- Validate required top-level keys: `output_root`, `common_params`, `problem`, `architectures`, `methods`, and `plotting`.
- Validate `problem.functional.kind` is exactly one of `KL`, `MMD`, or `CrossEntropy`.
- Implement architecture grid expansion.
- Implement method grid expansion using `methods: list[dict]`.
- Validate method names against `str_to_method`.
- Merge `method_kwargs` as `{ **common_params, **expanded_method_kwargs }`.
- Implement deterministic planned-run generation with stable `run_index` and `run_id`.
- Implement single-run selection via `--run-id`.
- Implement master/worker execution with `parallel.n_workers`.
- Ensure each worker writes `expanded_config.json` before execution.
- Ensure each worker writes and updates `status.json`.
- Ensure failures write `error.txt` with error message and traceback.
- Ensure successful runs write arrays/results into per-run output directories.
- Ensure successful runs write both checkpoints: `checkpoints/best/` and `checkpoints/last/`.
- Use Orbax `StandardCheckpointer` for full NNX state checkpointing.
- Implement `load_completed_runs(session_dir)`.
- Implement `load_model_checkpoint(session_dir, run_id, key="last")`.
- Implement `load_experiment_entries(session_dir)` returning plain per-run dictionaries with unpacked arrays and checkpoint paths.
- Implement `entries_to_frame(entries)` plus grouping helpers for architecture, method, and restart groups.
- Implement `get_lines(entries, style_channels)` with architecture grouping and restart aggregation.
- Implement per-run diagnostic plots (`plot_run_diagnostics`, `generate_per_run_plots`).
- Implement aggregate loss plots (`generate_aggregate_plots`) with mean +/- std tubes.
- Implement plot-only mode using the Loading API.
- Ensure workers only write inside unique per-run output directories.
- Ensure aggregate plotting or summaries are generated by the master after workers finish.
- Ensure CLI exits with code 0 on success and code 1 on error.
- Document project-specific choices: checkpoint format, best-checkpoint criterion, array file format, plot outputs, restart aggregation behavior, and behavior when no successful runs are available.

Recommended smoke tests:

- Config validation rejects missing required keys.
- Config validation rejects unsupported functional kinds.
- Config validation rejects unknown method names.
- Grid expansion produces expected number and order of planned runs.
- Method-specific kwargs override `common_params`.
- `--run-id` selects the correct planned run.
- Failed runs write `status.json` and `error.txt`.
- Successful runs write `expanded_config.json`, arrays, and both checkpoints.
- `load_completed_runs()` skips failed runs and loads successful arrays.
- `load_model_checkpoint()` loads `best` and `last`.
- `methods[].n_restarts` expands into the expected number of planned runs with deterministic `restart_index` values.
- Runs that differ only by `restart_index` aggregate into mean +/- std restart tubes.
- Architecture differences create separate architecture groups.
- Aggregate plot filenames: `aggregate_loss.pdf` for no architecture variation, `aggregate_loss_arch_XXX.pdf` per group.
- Plot-only mode regenerates per-run and aggregate plots from saved results.
- CLI exits with code 0 on success.
