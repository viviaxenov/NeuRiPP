# Batch-size selection — cifar10 / none / UNet-small

Dataset: `cifar10` · Architecture: UNet small
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `cifar10_unet_small_gpu1_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 50 | 29976 | 599.53 | 5.7 | 0.14 | False |
| 100 **| 59332 **| 593.32 **| 9.7 **| 0.24 **| False **|
| 300 | 176062 | 586.87 | 31.1 | 0.78 | False |
| 600 | OOM | OOM | OOM | — | — |
| 1000 | OOM | OOM | OOM | — | — |
| 2000 | OOM | OOM | OOM | — | — |
| 3000 | OOM | OOM | OOM | — | — |

**Chosen (1 GPUs): global 100, per-GPU 100** — peak 9.7 GiB (0.24).
Notes: 600: OOM/failed; 1000: OOM/failed; 2000: OOM/failed; 3000: OOM/failed

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
