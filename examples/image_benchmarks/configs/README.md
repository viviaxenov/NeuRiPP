# Image benchmark presets

See the main [image benchmark guide](../README.md) for installation, datasets,
local storage paths, architectures, external assets, and reproducibility rules.

Each JSON file is dedicated to `flow_matching_image_benchmark_runner.py` and
launches directly with:

```bash
python examples/flow_matching_image_benchmark_runner.py \
  --config examples/image_benchmarks/configs/<file>
```

Preset files may use `extends` only to avoid duplicating common training and
evaluation fields. The runner resolves inheritance before validation and saves
the complete resolved JSON in every session.

The DiffuseNNX `vae_trial1.pkl` has no trusted checksum bundled with NeuRiPP.
VAE presets therefore contain a 64-zero placeholder for `expected_sha256`.
Replace it with a trusted, independently obtained checksum before running; the
runner intentionally rejects a checkpoint that does not match it.

Fashion-MNIST presets are `fashion_mnist_mlp.json`,
`fashion_mnist_ae64_mlp.json`, and `fashion_mnist_unet.json`. The additional
`fashion_mnist_unet_ngd_smoke.json` is a two-GPU, global-batch-512 integration
check that runs exactly ten NGD optimizer steps with FID/KID disabled and MMD
plus sliced-Wasserstein enabled.

`fashion_mnist_unet_300epoch_adamw_ngd.json` compares AdamW and constant-step
NGD on the same Fashion-MNIST U-Net, seeds, and batches. With 60,000 training
examples and batch 2000 (a divisor of 60,000), each loader epoch is exactly 30
updates with no dropped images; therefore 300 complete loader epochs are 9,000
optimizer steps (18,000,000 examples processed, exactly 300
dataset-size-normalized effective epochs). The methods run in parallel, one GPU
each (GPUs 0 and 1); NGD uses at most 50 CG iterations per update with its
requested constant step and regularization. The comparison uses the compact
U-Net (`base_channels=16`, multipliers `[1,2]`, one residual block),
identically for both optimizers.

Batch 2000 is the largest divisor of 60,000 that fits one A100-40GB with
headroom: an NGD-only single-GPU memory sweep (`probe_ngd_memory.py`,
candidates 600-4000, CG maxiter 50) measured a stable ~21 GiB peak at batch
2000, while batch 3000 ran at the BFC arena ceiling (~31 GiB) and batch 4000
failed with `RESOURCE_EXHAUSTED` (a single 29.7 GiB allocation).
