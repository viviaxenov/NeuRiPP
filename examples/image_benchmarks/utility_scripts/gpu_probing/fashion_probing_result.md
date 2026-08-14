# Fashion-MNIST U-Net GPU probing result

Single-GPU NGD memory/throughput sweep for the Fashion-MNIST compact U-Net
comparison on `escher-02` (A100-40GB), run 2026-08-14. Raw data:
`fashion_probing_results/` (per-candidate `batch_*.json` and `summary.json`).

## Setup

- Model: compact U-Net, `base_channels=16`, `channel_mult=[1, 2]`,
  `num_res_blocks=1` (163,985 parameters), float32.
- Method: NGD, constant `step_size=1e-3`, `linear_solver_regularization=1e-3`,
  `linear_solver_tolerance=1e-6`, **CG `linear_solver_maxiter=50`**
  (worst-case memory path; AdamW has lower memory than NGD and was not probed).
- GPUs: 1 per candidate (`--gpu-count 1`), one fresh process per candidate
  (`--warmup 8 --measure 20`).
- Config: `fashion_mnist_unet_300epoch_adamw_ngd.json`.
- Candidates: divisors of the 60,000-image training split.
- Measurement: pynvml `memory.used` high-water with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (grow-on-demand BFC arena, default
  `MEM_FRACTION`), one `device_memory_profile()` pprof snapshot after the
  window, per-step wall time.

## Results

| batch | peak mem (GiB) | frac of 40 GiB | s/step | ms/sample | SM% mean | fit |
|---:|---:|---:|---:|---:|---:|---|
| 600 | 10.4 | 0.26 | 1.59 | 2.66 | 91 | ✅ |
| 1000 | 19.4 | 0.49 | 2.56 | 2.56 | 92 | ✅ |
| 1200 | 18.9 | 0.47 | 3.03 | 2.53 | 92 | ✅ |
| 1500 | 19.4 | 0.49 | 3.74 | 2.50 | 92 | ✅ |
| **2000** | **21.4** | **0.54** | **4.96** | **2.48** | **92** | ✅ **chosen** |
| 3000 | 31.0 | 0.78 | 7.30 | 2.43 | 91 | ⚠️ at arena ceiling |
| 4000 | — | — | — | — | — | ❌ `RESOURCE_EXHAUSTED` (29.69 GiB single allocation) |

Notes:
- Peak memory is the grow-on-demand high-water, a truthful upper bound of the
  per-step transient. `device_memory_used_steady_growth_in_window` was
  `false` for every completed candidate (peaks stable across the window).
- `pprof_live_bytes` was 1,970,252 bytes (≈1.9 MB) for every candidate —
  persistent parameter state only, no per-step leak.
- `ms/sample` is roughly constant (~2.4–2.7 ms/sample): steps are
  latency-bound, so batch size trades memory for number of steps, not
  throughput.
- Batch 3000 peaked at the BFC arena ceiling (0.75 fraction ≈ 29.6 GiB) — it
  completed but has no evaluation-headroom margin.

## Decision

**batch = 2000** — the largest divisor of 60,000 that fits one A100-40GB with
comfortable headroom:

- peak ≈ 21.4 GiB (54% of the device; 58% of the 0.9-fraction production arena
  of ~36 GiB), leaving room for the periodic evaluation workload (held-out FM
  loss, sampling, sample metrics);
- batch 4000 does not fit (29.69 GiB single allocation OOM);
- batch 3000 runs at the arena ceiling with no headroom.

Consequences for the comparison run (`fashion_mnist_unet_300epoch_adamw_ngd.json`):
- global batch 2000, `max_steps = 300 × (60000 / 2000) = 9000`,
- 18,000,000 examples processed, none dropped,
- methods run in parallel, one GPU each (`gpus_per_run 1`,
  `max_concurrent_runs 2`), expected wall time ≈ 12–13 h for NGD.

## Repro commands

```bash
# on escher-02, env neuripp_cuda13, repo checkout
mkdir -p probe_results && cd <repo>
python examples/image_benchmarks/utility_scripts/gpu_probing/probe_ngd_memory.py \
  --config examples/image_benchmarks/configs/fashion_mnist_unet_300epoch_adamw_ngd.json \
  --sweep --candidates 600,1000,1200,1500,2000,3000,4000 \
  --warmup 8 --measure 20 --output-dir probe_results --gpu-index 0 --gpu-count 1
```