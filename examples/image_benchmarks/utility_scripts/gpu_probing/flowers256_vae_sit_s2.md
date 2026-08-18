# Batch-size selection — flowers102 @256 / VAE / SiT-S/2

Dataset: `flowers102` · Architecture: SiT-S/2 @256
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `flowers256_vae_sit_s2_gpu2_results/`, `flowers256_vae_sit_s2_gpu4_results/`

## 2 GPUs (global batch sharded over 2 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 **| 5855 **| 182.97 **| 19.1 **| 0.48 **| False **|
| 64 | 11284 | 176.31 | 31.7 | 0.79 | False |
| 128 | 22127 | 172.87 | 31.7 | 0.79 | False |
| 256 | OOM | OOM | OOM | — | — |
| 512 | OOM | OOM | OOM | — | — |
| 1024 | OOM | OOM | OOM | — | — |

**Chosen (2 GPUs): global 32, per-GPU 16** — peak 19.1 GiB (0.48).
Notes: 256: OOM/failed; 512: OOM/failed; 1024: OOM/failed

## 4 GPUs (global batch sharded over 4 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 3136 | 97.99 | 11.1 | 0.28 | False |
| 64 **| 5875 **| 91.80 **| 19.1 **| 0.48 **| False **|
| 128 | 11302 | 88.29 | 31.7 | 0.79 | False |
| 256 | 22165 | 86.58 | 31.7 | 0.79 | False |
| 512 | OOM | OOM | OOM | — | — |
| 1024 | OOM | OOM | OOM | — | — |

**Chosen (4 GPUs): global 64, per-GPU 16** — peak 19.1 GiB (0.48).
Notes: 512: OOM/failed; 1024: OOM/failed

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
