# Batch-size selection — mnist / none / MLP

Dataset: `mnist` · Architecture: MLP
Tool: `probe_ngd_memory.py` (NGD worst-case path, `--warmup 8 --measure 20`, `PREALLOCATE=false`, arena 0.75)
Raw results: `mnist_mlp_gpu1_results/`

## 1 GPUs (global batch sharded over 1 device(s))

| batch size | ms/step | ms/image | peak mem (GiB) | frac | steady |
|---:|---:|---:|---:|---:|---:|
| 50 | 13 | 0.26 | 1.4 | 0.04 | False |
| 100 | 21 | 0.21 | 1.4 | 0.04 | False |
| 300 | 42 | 0.14 | 1.5 | 0.04 | False |
| 600 | 65 | 0.11 | 1.5 | 0.04 | False |
| 1000 | 87 | 0.09 | 1.6 | 0.04 | False |
| 2000 | 136 | 0.07 | 1.7 | 0.04 | False |
| 3000 **| 201 **| 0.07 **| 1.7 **| 0.04 **| False **|

**Chosen (1 GPUs): global 3000, per-GPU 3000** — peak 1.7 GiB (0.04).

---
Legend: ms/step = mean step wall time; ms/image = ms/step ÷ global batch; peak mem = `device_memory_peak_seen_max` (first GPU of the range); steady = `device_memory_used_steady_growth_in_window`.
