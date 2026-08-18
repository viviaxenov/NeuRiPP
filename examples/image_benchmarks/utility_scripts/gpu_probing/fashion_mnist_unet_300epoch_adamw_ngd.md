# Batch-size selection — fashion_mnist / none / UNet (16ch, 300ep)

Dataset: `fashion_mnist` · Architecture: UNet small
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `fashion_mnist_unet_300epoch_adamw_ngd_gpu1_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 50 | 1201 | 24.02 | 1.9 | 0.05 | False |
| 100 | 2279 | 22.79 | 2.4 | 0.06 | False |
| 300 | 6664 | 22.21 | 5.4 | 0.13 | False |
| 600 | 13264 | 22.11 | 9.4 | 0.24 | False |
| 1000 | 22033 | 22.03 | 17.4 | 0.44 | False |
| 2000 **| 43943 **| 21.97 **| 17.4 **| 0.44 **| False **|
| 3000 | 65975 | 21.99 | 31.0 | 0.78 | False |

**Chosen (1 GPUs): global 2000, per-GPU 2000** — peak 17.4 GiB (0.44).

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
