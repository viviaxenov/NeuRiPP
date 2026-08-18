# Batch-size selection — imagenet64 / none / SiT-S/2

Dataset: `imagenet64` · Architecture: SiT-S/2 @64
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `imagenet64_sit_s2_gpu1_results/`, `imagenet64_sit_s2_gpu2_results/`, `imagenet64_sit_s2_gpu4_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | OOM | OOM | OOM | — | — |
| 64 | OOM | OOM | OOM | — | — |
| 128 | OOM | OOM | OOM | — | — |
| 256 | OOM | OOM | OOM | — | — |
| 512 | OOM | OOM | OOM | — | — |
| 1024 | OOM | OOM | OOM | — | — |

Notes: 32: OOM/failed; 64: OOM/failed; 128: OOM/failed; 256: OOM/failed; 512: OOM/failed; 1024: OOM/failed

## 2 GPUs (global batch sharded over 2 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | OOM | OOM | OOM | — | — |
| 64 | OOM | OOM | OOM | — | — |
| 128 | OOM | OOM | OOM | — | — |
| 256 | OOM | OOM | OOM | — | — |
| 512 | OOM | OOM | OOM | — | — |
| 1024 | OOM | OOM | OOM | — | — |

Notes: 32: OOM/failed; 64: OOM/failed; 128: OOM/failed; 256: OOM/failed; 512: OOM/failed; 1024: OOM/failed

## 4 GPUs (global batch sharded over 4 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 **| 15006 **| 468.95 **| 31.7 **| 0.79 **| False **|
| 64 | OOM | OOM | OOM | — | — |
| 128 | OOM | OOM | OOM | — | — |
| 256 | OOM | OOM | OOM | — | — |

**Chosen (4 GPUs): global 32, per-GPU 8** — peak 31.7 GiB (0.79).
Notes: 64: OOM/failed; 128: OOM/failed; 256: OOM/failed

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
