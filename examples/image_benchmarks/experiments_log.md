# Fashion-MNIST Experiments

| Date | Config | Summary |
|---|---|---|
| 2026-08-13 | `fashion_mnist_unet_300epoch_adamw_ngd.json` | Established the Fashion-MNIST AdamW/NGD compact U-Net comparison and sample-metric setup. |
| 2026-08-14 | `fashion_mnist_unet_300epoch_adamw_ngd.json` | Ran the batch-2000 optimizer comparison and GPU batch-size probing. |
| 2026-08-19 | `fashion_mnist_unet_300epoch_adamw_ngd.json` | Added benchmark plotting, fixed validation evaluation, EMA support, and NGD sweep infrastructure. |
| 2026-08-20 | `fashion_mnist_unet_300epoch_adamw_ngd.json` | Added metric-history persistence and applied GPU-probed batch sizing. |
| 2026-08-22 | `fashion_mnist_ae64_ngd_sweep.json` | Ran the broad Fashion-MNIST AE-64 NGD hyperparameter sweep. |
| 2026-08-24 | `fashion_mnist_ae64_ngd_adam_equal_train_time.json` | Compared the best NGD configuration against AdamW at matched training wall-clock time. |
| 2026-08-26 | `fashion_mnist_ae64_ngd_matvec_batch_sweep.json` | Tested CG limits `10/50/100` with matvec batches `300/1500/3000`. |
| 2026-08-27 | `fashion_mnist_ae64_ngd_sweep_matvec_adam_comparison.json` | Re-tested all NGD sweep parameters with matvec batches `3000/300`, plus the best AdamW baseline. |
| 2026-09-09 | `cifar10_facebook_2_07_review.json` | Added the functional Guided-Diffusion-style Facebook CIFAR-10 U-Net preset, scale-shift residual blocks, guided QKV attention, convolution-free resampling, skewed EDM timestep sampling, mean-all-elements loss reduction, and lightweight evaluation checkpoints. |
| 2026-09-09 | `cifar10_facebook_2_07_review.json` | Probed global batch 64 on 1/2/4/8 NVIDIA A100-SXM4-40GB GPUs for NGD and AdamW. NGD required at least 2 GPUs; AdamW fit on 1 GPU. |

## Findings So Far

- Good NGD parameters were found: `step_size=0.01`, regularization `0.1`, CG limit `50`, tolerance `1e-6`, and clipping threshold `10`.
- On Fashion-MNIST, a `1/10` matvec subsample (`300` versus gradient batch `3000`) matched or slightly outperformed the full-matvec comparison in wall-clock time at this stage.

## CIFAR-10 Facebook U-Net Probing — 2026-09-09

Host: `escher-02` · 8x NVIDIA A100-SXM4-40GB · conda environment `neuripp_cuda13`

The Facebook-compatible U-Net contains `55,676,419` parameters. Probes used global batch 64, full-batch NGD metric computation where applicable, `--warmup 8 --measure 20`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Memory and utilization are sampled from the first GPU in each run.

### NGD

| GPUs | Batch/GPU | Status | ms/step | ms/sample | Peak GiB | Memory fraction | SM mean |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 64 | OOM: 18.08 GiB allocation failure | — | — | — | — | — |
| 2 | 32 | Pass | 4998.7 | 78.1051 | 28.13 | 70.35% | 98.06% |
| 4 | 16 | Pass | 3224.5 | 50.3832 | 17.14 | 42.84% | 95.51% |
| 8 | 8 | Pass | 2357.1 | 36.8303 | 12.13 | 30.34% | 91.94% |

### AdamW

| GPUs | Batch/GPU | Status | ms/step | ms/sample | Peak GiB | Memory fraction | SM mean |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 64 | Pass | 170.2 | 2.6598 | 13.41 | 33.52% | 58.42% |
| 2 | 32 | Pass | 119.3 | 1.8639 | 18.06 | 45.15% | 45.52% |
| 4 | 16 | Pass | 112.1 | 1.7515 | 18.06 | 45.14% | 31.83% |
| 8 | 8 | Pass | 145.3 | 2.2706 | 12.06 | 30.15% | 26.03% |
