# Batch-size selection — ffhq64 / none / UNet-CIFAR-reference

Dataset: `ffhq64` · Architecture: UNet CIFAR-reference @64
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `ffhq64_unet_gpu1_results/`, `ffhq64_unet_gpu2_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 50 **| 440648 **| 8812.96 **| 31.1 **| 0.78 **| False **|
| 100 | OOM | OOM | OOM | — | — |
| 300 | OOM | OOM | OOM | — | — |
| 600 | OOM | OOM | OOM | — | — |
| 1000 | OOM | OOM | OOM | — | — |
| 2000 | OOM | OOM | OOM | — | — |
| 3000 | OOM | OOM | OOM | — | — |

**Chosen (1 GPUs): global 50, per-GPU 50** — peak 31.1 GiB (0.78).
Notes: 100: OOM/failed; 300: OOM/failed; 600: OOM/failed; 1000: OOM/failed; 2000: OOM/failed; 3000: OOM/failed

## 2 GPUs (global batch sharded over 2 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 50 **| 221566 **| 4431.33 **| 19.1 **| 0.48 **| False **|
| 100 | 441117 | 4411.17 | 31.7 | 0.79 | False |
| 300 | OOM | OOM | OOM | — | — |
| 600 | OOM | OOM | OOM | — | — |
| 1000 | OOM | OOM | OOM | — | — |
| 2000 | OOM | OOM | OOM | — | — |
| 3000 | OOM | OOM | OOM | — | — |

**Chosen (2 GPUs): global 50, per-GPU 25** — peak 19.1 GiB (0.48).
Notes: 300: OOM/failed; 600: OOM/failed; 1000: OOM/failed; 2000: OOM/failed; 3000: OOM/failed

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
