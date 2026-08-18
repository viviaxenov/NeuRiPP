# Batch-size selection — cifar10 / none / UNet-CIFAR-reference

Dataset: `cifar10` · Architecture: UNet CIFAR-reference
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `cifar10_unet_gpu1_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 50 **| 115484 **| 2309.68 **| 18.5 **| 0.46 **| False **|
| 100 | 228796 | 2287.96 | 31.1 | 0.78 | False |
| 300 | OOM | OOM | OOM | — | — |
| 600 | OOM | OOM | OOM | — | — |
| 1000 | OOM | OOM | OOM | — | — |
| 2000 | OOM | OOM | OOM | — | — |
| 3000 | OOM | OOM | OOM | — | — |

**Chosen (1 GPUs): global 50, per-GPU 50** — peak 18.5 GiB (0.46).
Notes: 300: OOM/failed; 600: OOM/failed; 1000: OOM/failed; 2000: OOM/failed; 3000: OOM/failed

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
