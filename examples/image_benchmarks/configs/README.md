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
