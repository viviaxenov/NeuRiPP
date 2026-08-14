# GPU probing protocol

Measure the real GPU memory footprint and throughput of the image-benchmark
training path at different **global batch sizes**, so that the largest global
batch that fits a given device / GPU count with headroom can be chosen. The
probe runs the *same* production code path (`FlowMatching` wrapper,
`ImageTrainer`, NGD with CG `linear_solver_maxiter=50`), so the measured
numbers are the ones that matter for the real runs.

## When to use

Pick a global batch size for a new experiment (different dataset, different
architecture/rhs, or a different number of GPUs) on `escher-02` (8x A100-40GB).
Choose batch sizes that **divide the training split** so no images are dropped
and each loader epoch has an integer number of updates.

## Tool

- Script: `probe_ngd_memory.py` (this directory)
- Config: any image-benchmark preset — `--config` selects the **dataset**
  (through the preset's `problem.dataset`) and the **architecture** (through
  `rhs`). The probe overrides the method with the worst-case NGD memory path.
- GPU count: `--gpu-count N` shards each global batch over N GPUs
  (default 1). `--gpu-index` selects the first physical GPU of the range.
- Host / environment:
  - remote host `escher-02`,
  - conda env `neuripp_cuda13`,
  - repo checkout at
    `/Home/optimier/aksenov/Documents/code/scratch/neuripp_fashion_300epoch`,
  - dataset cache present in that checkout.

## Measurement rationale

- **Peak memory (`device_memory_peak_seen_max`)** is the max pynvml
  `memory.used` over the measurement window with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The BFC arena then grows on demand to
  cover the largest per-step transient, so this high-water is a **truthful
  upper bound** of the per-step peak. A preallocated arena
  (`XLA_PYTHON_CLIENT_MEM_FRACTION`) would pin `memory.used` to a constant and
  hide the transient; that is why the probe disables preallocation.
- **OOM is definitive**: a hard `RESOURCE_EXHAUSTED` during the window means
  the candidate does not fit. There is no substitute for running the real step.
- **`device_memory_used_steady_growth_in_window`**: if `memory.used` is still
  climbing in the last third of the window, the seen high-water may
  underestimate the true peak (the window was too short / the candidate is
  borderline).
- **`pprof_live_bytes`**: a single `jax.profiler.device_memory_profile()`
  snapshot taken *after* the window. It only counts persistent JAX arrays
  (~parameter state, here ~1.9 MB) and is used as a **leak detector**, not as
  the transient peak. It is taken once, not per step: per-step profile calls
  disable XLA buffer reuse and distort the grow-on-demand arena.
- **One candidate per process**: the sweep spawns a fresh process per batch
  size so arena state never leaks between candidates and an OOM is attributable
  to that single candidate.
- **Throughput**: `mean_step_seconds` (steps block until ready) and
  `time_per_sample_ms = mean_step_seconds / batch_size * 1000`. Because steps
  are latency-bound, `time_per_sample_ms` is roughly constant across batch
  sizes; the batch choice therefore trades memory for wall-clock steps, not
  throughput.
- **SM utilization** (`pynvml_sm_percent_max/mean`) is sampled on the first
  GPU of the range; for data-parallel runs the per-GPU footprint is identical
  across replicas.

## Environment rules (set before any JAX import)

The script derives these itself from the CLI; do not set them by hand unless
debugging:

```
CUDA_VISIBLE_DEVICES     = gpu_index .. gpu_index+gpu_count-1
XLA_PYTHON_CLIENT_PREALLOCATE = false
XLA_PYTHON_CLIENT_MEM_FRACTION = (unset -> default 0.75)
XLA_PYTHON_CLIENT_ALLOCATOR    = (unset -> BFC)
TF_GPU_ALLOCATOR               = (unset)
```

Do **not** use `MEM_FRACTION=1.0` (the arena balloons to the full device during
cuDNN/runtime autotune and produces spurious OOMs), and do **not** use the
`platform` allocator (stream-capture dealloc failures in JAX 0.11).

## Candidate selection

Pick candidates that:

1. divide the training split (the probe rejects other batch sizes with
   "batch size does not divide the training split"), and
2. are divisible by `--gpu-count`.

Example for Fashion-MNIST (60,000 training examples, 1 GPU):
`600, 1000, 1200, 1500, 2000, 3000, 4000`.

## Workflow

1. Deploy: the probe is versioned in the repo, so `git pull` on the remote
   checkout is enough (no scp).
2. Launch the sweep in the background (one process per candidate):

```bash
cd <repo> && mkdir -p probe_results
CUDA_VISIBLE_DEVICES=<first-gpu> nohup python \
  examples/image_benchmarks/utility_scripts/gpu_probing/probe_ngd_memory.py \
  --config examples/image_benchmarks/configs/<preset>.json \
  --sweep --candidates <a,b,c,...> --warmup 8 --measure 20 \
  --output-dir probe_results --gpu-index <idx> --gpu-count <N> \
  > probe_sweep.log 2>&1 &
```

   Typical settings: `--warmup 8` (compilation + cuDNN autotune),
   `--measure 20` (steady window for peak and step time).

   For a **single candidate** (not a sweep), files are written via
   `--output <path>` (`--output-dir` only applies to `--sweep` mode):

```bash
python examples/image_benchmarks/utility_scripts/gpu_probing/probe_ngd_memory.py \
  --config examples/image_benchmarks/configs/<preset>.json \
  --batch <batch> --warmup 8 --measure 20 \
  --output probe_results/batch_<batch>.json \
  --gpu-index <idx> --gpu-count <N>
```

3. Monitor while it runs:
   - `tail probe_sweep.log` (prints `[sweep] batch_XXXXX.json: rc=... peak=...`),
   - `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`,
   - make sure the candidate process is alive (`ps aux | grep probe_ngd_memory`).

4. Collect results: `probe_results/summary.json` plus one `batch_XXXXX.json`
   per candidate. rsync them to the machine where the report is written.

5. Render the result table:

| batch | peak mem (GiB) | s/step | ms/sample | SM% mean | fit |
|---|---|---|---:|---:|---|

   `GiB = device_memory_peak_seen_max / 2^30`,
   `ms/sample = mean_step_seconds / batch_size * 1000`.

## Decision rule

Choose the **largest candidate** that:
- completed without OOM,
- had `device_memory_used_steady_growth_in_window == false` (peak was stable),
- leaves comfortable headroom below the production arena fraction (0.9, i.e.
  ~36 GiB on an A100-40GB) — enough also for the periodic evaluation workload
  (held-out FM loss, sampling, MMD / sliced Wasserstein) that runs in the real
  training loop but is not part of the probe window.

If the two largest fitting candidates differ mainly in memory (not
throughput), prefer the one with the larger headroom. Record the raw
`summary.json` / `batch_*.json` next to the result report.

## Pitfalls learned (JAX 0.11)

- `XLA_PYTHON_CLIENT_ALLOCATOR=platform` -> `CUDA_ERROR_STREAM_CAPTURE`
  dealloc errors during compilation; use the default BFC allocator.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=1.0` -> arena balloons to the full device
  during autotune; spurious OOMs. Use the default 0.75.
- Preallocation (`PREALLOCATE` unset/true) pins `memory.used` to a constant and
  hides the transient peak; the probe disables it.
- Calling `jax.profiler.device_memory_profile()` every step disables XLA buffer
  reuse and inflates the grow-on-demand arena; call it once, after the window.
- `jax.devices()[0].memory_stats()` is not implemented in JAX 0.11; the pprof
  proto (gzip'd) is the only in-process profile, and it only counts persistent
  arrays.
- Batch sizes that do not divide the training split are rejected by the probe;
  non-power-of-2 divisors (e.g. 600, 1000, 2000) have at most low single-digit
  percent throughput impact and are fine.
