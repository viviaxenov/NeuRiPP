# Batch-size selection — flowers102 @256 / VAE / latent-UNet

Dataset: `flowers102` · Architecture: latent-UNet @256
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `flowers256_vae_unet_gpu2_results/`, `flowers256_vae_unet_gpu4_results/`

## 2 GPUs (global batch sharded over 2 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 10877 | 339.91 | 5.1 | 0.13 | False |
| 64 | 19920 | 311.26 | 7.1 | 0.18 | False |
| 128 | 38624 | 301.75 | 11.1 | 0.28 | False |
| 256 **| 75915 **| 296.54 **| 19.2 **| 0.48 **| False **|
| 512 | 150386 | 293.72 | 31.7 | 0.79 | False |
| 1024 | OOM | OOM | OOM | — | — |

**Chosen (2 GPUs): global 256, per-GPU 128** — peak 19.2 GiB (0.48).
Notes: 1024: OOM/failed

## 4 GPUs (global batch sharded over 4 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 6337 | 198.03 | 4.1 | 0.10 | False |
| 64 | 10903 | 170.36 | 5.1 | 0.13 | False |
| 128 | 19966 | 155.99 | 7.1 | 0.18 | False |
| 256 | 38650 | 150.98 | 11.2 | 0.28 | False |
| 512 **| 75993 **| 148.42 **| 19.2 **| 0.48 **| False **|
| 1024 | 150500 | 146.97 | 31.7 | 0.79 | False |

**Chosen (4 GPUs): global 512, per-GPU 128** — peak 19.2 GiB (0.48).

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
