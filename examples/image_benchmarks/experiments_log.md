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

## Findings So Far

- Good NGD parameters were found: `step_size=0.01`, regularization `0.1`, CG limit `50`, tolerance `1e-6`, and clipping threshold `10`.
- On Fashion-MNIST, a `1/10` matvec subsample (`300` versus gradient batch `3000`) matched or slightly outperformed the full-matvec comparison in wall-clock time at this stage.
