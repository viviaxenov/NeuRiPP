# Batch-size selection — lsun_church @256 / VAE / latent-UNet

Dataset: `lsun_church` · Architecture: latent-UNet @256
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `lsun256_vae_unet_gpu2_results/`, `lsun256_vae_unet_gpu4_results/`

## 2 GPUs (global batch sharded over 2 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 10853 | 339.15 | 5.1 | 0.13 | False |
| 64 | 19890 | 310.79 | 7.1 | 0.18 | False |
| 128 | 38620 | 301.72 | 11.1 | 0.28 | False |
| 256 **| 75871 **| 296.37 **| 19.2 **| 0.48 **| False **|
| 512 | 150362 | 293.68 | 31.7 | 0.79 | False |
| 1024 | OOM | OOM | OOM | — | — |

**Chosen (2 GPUs): global 256, per-GPU 128** — peak 19.2 GiB (0.48).
Notes: 1024: OOM/failed

## 4 GPUs (global batch sharded over 4 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 32 | 6297 | 196.77 | 4.1 | 0.10 | False |
| 64 | 10889 | 170.14 | 5.1 | 0.13 | False |
| 128 | 19912 | 155.56 | 7.1 | 0.18 | False |
| 256 | 38632 | 150.91 | 11.2 | 0.28 | False |
| 512 **| 75893 **| 148.23 **| 19.2 **| 0.48 **| False **|
| 1024 | 150436 | 146.91 | 31.7 | 0.79 | False |

**Chosen (4 GPUs): global 512, per-GPU 128** — peak 19.2 GiB (0.48).

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
