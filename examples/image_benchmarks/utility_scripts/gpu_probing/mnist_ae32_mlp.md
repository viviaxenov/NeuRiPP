# Batch-size selection — mnist / AE-32 / MLP

Dataset: `mnist` · Architecture: MLP on AE-32 latent
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `mnist_ae32_mlp_gpu1_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 50 | 12 | 0.24 | 1.4 | 0.03 | False |
| 100 | 14 | 0.14 | 1.4 | 0.03 | False |
| 300 | 22 | 0.07 | 1.4 | 0.04 | False |
| 600 | 32 | 0.05 | 1.5 | 0.04 | False |
| 1000 | 46 | 0.05 | 1.5 | 0.04 | False |
| 2000 | 77 | 0.04 | 1.6 | 0.04 | False |
| 3000 **| 111 **| 0.04 **| 1.6 **| 0.04 **| False **|

**Chosen (1 GPUs): global 3000, per-GPU 3000** — peak 1.6 GiB (0.04).

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
