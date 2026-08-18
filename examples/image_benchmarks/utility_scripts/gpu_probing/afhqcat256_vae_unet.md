# Batch-size selection — afhq_cat @256 / VAE / latent-UNet

Dataset: `afhq_cat` · Architecture: latent-UNet @256
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `afhqcat256_vae_unet_gpu2_results/`, `afhqcat256_vae_unet_gpu4_results/`

## 2 GPUs (global batch sharded over 2 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 10894 | 340.43 | 5.1 | 0.13 | False |
| 64 | 19949 | 311.71 | 7.1 | 0.18 | False |
| 128 | 38688 | 302.25 | 11.1 | 0.28 | False |
| 256 **| 75912 **| 296.53 **| 19.2 **| 0.48 **| False **|
| 512 | 150491 | 293.93 | 31.7 | 0.79 | False |
| 1024 | OOM | OOM | OOM | — | — |

**Chosen (2 GPUs): global 256, per-GPU 128** — peak 19.2 GiB (0.48).
Notes: 1024: OOM/failed

## 4 GPUs (global batch sharded over 4 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 6327 | 197.71 | 4.1 | 0.10 | False |
| 64 | 10908 | 170.43 | 5.1 | 0.13 | False |
| 128 | 19960 | 155.94 | 7.1 | 0.18 | False |
| 256 | 38653 | 150.99 | 11.2 | 0.28 | False |
| 512 **| 75970 **| 148.38 **| 19.2 **| 0.48 **| False **|
| 1024 | 150513 | 146.99 | 31.7 | 0.79 | False |

**Chosen (4 GPUs): global 512, per-GPU 128** — peak 19.2 GiB (0.48).

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
