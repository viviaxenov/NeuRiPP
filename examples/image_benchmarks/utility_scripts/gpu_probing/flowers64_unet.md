# Batch-size selection — flowers102 @64 / none / UNet-small

Dataset: `flowers102` · Architecture: UNet small @64
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `flowers64_unet_gpu1_results/`, `flowers64_unet_gpu2_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 73145 | 2285.79 | 9.7 | 0.24 | False |
| 64 **| 144804 **| 2262.56 **| 17.7 **| 0.44 **| False **|
| 128 | 288672 | 2255.25 | 31.1 | 0.78 | False |
| 256 | OOM | OOM | OOM | — | — |
| 512 | OOM | OOM | OOM | — | — |
| 1024 | OOM | OOM | OOM | — | — |

**Chosen (1 GPUs): global 64, per-GPU 64** — peak 17.7 GiB (0.44).
Notes: 256: OOM/failed; 512: OOM/failed; 1024: OOM/failed

## 2 GPUs (global batch sharded over 2 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 37311 | 1165.97 | 6.4 | 0.16 | False |
| 64 | 73188 | 1143.56 | 10.4 | 0.26 | False |
| 128 **| 144891 **| 1131.96 **| 18.4 **| 0.46 **| False **|
| 256 | 288631 | 1127.47 | 31.7 | 0.79 | False |
| 512 | OOM | OOM | OOM | — | — |
| 1024 | OOM | OOM | OOM | — | — |

**Chosen (2 GPUs): global 128, per-GPU 64** — peak 18.4 GiB (0.46).
Notes: 512: OOM/failed; 1024: OOM/failed

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
