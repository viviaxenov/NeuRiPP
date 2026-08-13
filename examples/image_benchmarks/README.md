# Image flow-matching benchmarks

This directory contains NeuRiPP's reproducible, unconditional image-generation
benchmarks. The harness compares the project's optimization methods while
holding dataset preparation, representation, vector-field architecture, random
seeds, and evaluation fixed.

## Installation

Install the complete benchmark stack, including the pinned installable
DiffuseNNX fork, with:

```bash
python -m pip install -e ".[image-benchmarks-diffuse]"
```

Add `cuda12` or `cuda13` for the matching JAX accelerator build, for example:

```bash
python -m pip install -e ".[cuda12,image-benchmarks-diffuse]"
```

The DiffuseNNX dependency is pinned to commit
[`da5f2b79497722931d279b012c90bec61050466b`](https://github.com/viviaxenov/diffuse_nnx/tree/da5f2b79497722931d279b012c90bec61050466b).

## Datasets and local paths

All datasets are obtained through
[`datasets.load_dataset`](https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset).
The supplied presets use `data/huggingface` at the repository root as the Hugging
Face cache. Change `problem.dataset.cache_dir` to use another location. Hugging
Face controls the internal dataset directory names beneath that cache.

| Registry name | Hugging Face dataset | Supported resolution | Split policy |
|---|---|---:|---|
| `mnist` | [`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist) | 28 | Reserve 5,000 training examples for validation; retain the provided test set. |
| `cifar10` | [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) | 32 | Reserve 5,000 training examples for validation; retain the provided test set. |
| `flowers102` | [`pufanyi/flowers102`](https://huggingface.co/datasets/pufanyi/flowers102) | 64, 256 | Use the provided train, validation, and test splits. |
| `afhq_cat` | [`bitmind/AFHQ`](https://huggingface.co/datasets/bitmind/AFHQ) | 256, 512 | Select `train/cat/*` and `test/cat/*`; reserve 10% of cat training images for validation. |
| `lsun_church` | [`tglcourse/lsun_church_train`](https://huggingface.co/datasets/tglcourse/lsun_church_train) | 256 | Reserve 5,000 training examples for validation; retain the provided test set. |
| `ffhq64` | [`Dmini/FFHQ-64x64`](https://huggingface.co/datasets/Dmini/FFHQ-64x64) | 64 | Deterministic 60,000/5,000/5,000 train/validation/test partition. |
| `imagenet64` | [`benjamin-paine/imagenet-1k-64x64`](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64) | 64 | Official validation is the final reference; reserve 5,000 training examples for FM validation. |
| `imagenet256` | [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) | 256 | Official validation is the final reference; reserve 5,000 training examples for FM validation. |

`imagenet256` is gated. Accept the dataset terms on Hugging Face and export
`HF_TOKEN` before preparation. The harness fails rather than substituting a
different dataset when access is missing. Once downloaded, set
`problem.dataset.offline=true` to reuse only local files.

NeuRiPP records resolved Hugging Face revisions and deterministic logical split
indices under:

```text
data/huggingface/neuripp_manifests/<dataset>/<digest>/
```

Images are center-square cropped, resized with Pillow Lanczos, represented as
NHWC `float32` in `[-1, 1]` during training, and converted to NHWC `uint8` for
FID/KID evaluation. Horizontal flips are configurable for pixel-space training
but disabled when latents are cached.

## Models and reference implementations

### Representations

- **Raw pixels (`none`)** use an identity encoder.
- **Project autoencoder (`ae`)** uses the local single-layer vector
  [autoencoder](encoders/project_ae_model.py), with configurable latent size and
  optional training when no checkpoint exists.
- **Stable-Diffusion-style VAE (`vae`)** adapts the pinned DiffuseNNX
  [`StabilityVAE`](https://github.com/viviaxenov/diffuse_nnx/blob/da5f2b79497722931d279b012c90bec61050466b/src/diffuse_nnx/networks/encoders/sd_vae.py).
  It supports deterministic posterior means or explicit RNG-controlled samples.

### Vector fields

- **Time-conditioned MLP** is the local Flax NNX
  [implementation](rhs/mlp.py), with sinusoidal time embeddings and an explicit
  flatten/unflatten adapter for controlled image experiments.
- **U-Net** is the local Flax NNX [implementation](rhs/unet.py), following the
  architecture semantics of TorchCFM's
  [`UNetModelWrapper`](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/unet/unet.py).
  Presets are `small`, `cifar_reference`, and `large`.
- **SiT/DiT** adapts the pinned DiffuseNNX
  [`DiT`](https://github.com/viviaxenov/diffuse_nnx/blob/da5f2b79497722931d279b012c90bec61050466b/src/diffuse_nnx/networks/transformers/dit_nnx.py)
  as an unconditional NeuRiPP RHS. Variants are `S`, `B`, `L`, and `XL`.

FID features use DiffuseNNX's
[`InceptionV3`](https://github.com/viviaxenov/diffuse_nnx/blob/da5f2b79497722931d279b012c90bec61050466b/src/diffuse_nnx/eval/inception.py).

## External assets

The benchmark can prepare assets at these default repository-relative paths:

| Asset | Default path | Preparation |
|---|---|---|
| Stable Diffusion VAE | `artifacts/encoders/vae_trial1.pkl` | Optional GCS download; requires a separately trusted checksum in the preset. |
| Inception FID weights | `artifacts/inception/inception_v3_weights_fid.pickle` | Verified HTTP download with the built-in historical checksum. |
| Project AE checkpoints | `artifacts/ae/<experiment>` | Trained locally when `train_if_missing=true`. |
| Cached VAE/AE latents | `artifacts/latents` | Generated from the prepared dataset and encoder. |
| FID features/statistics | `artifacts/fid` | Generated on first evaluation and reused by provenance key. |
| Runs and checkpoints | `results/image_benchmarks/<run>` | Created by the benchmark runner. |

The VAE and Inception files are Python pickles. They are never loaded before
their SHA-256 checks pass. The historical Inception checksum is:

```text
4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77
```

Automatic VAE download uses Google Cloud Storage and therefore requires Google
application-default credentials. The historical `vae_trial1.pkl` has no trusted
checksum bundled with NeuRiPP; replace the all-zero placeholder in VAE presets
with a digest obtained independently before running them.

## Running a benchmark

Choose a preset from [`configs/`](configs/README.md), then run:

```bash
python examples/flow_matching_image_benchmark_runner.py \
  --config examples/image_benchmarks/configs/cifar10_unet.json
```

Preset inheritance is resolved before execution, and the complete resolved
configuration is saved in the session directory. The runner supports resumable
checkpoints and assigns disjoint GPU groups to concurrent workers through
`CUDA_VISIBLE_DEVICES`; do not set that variable in `resources.worker_env`.

For a cheap end-to-end check, use `mnist_mlp_smoke.json`. Large presets can
download substantial datasets and produce large latent/FID caches.

## Reproducibility

The harness records the NeuRiPP commit, exact DiffuseNNX commit, Hugging Face
revision, dataset manifest digest, split-index checksums, model/asset checksums,
RNG seeds, and device environment. Cache keys include preprocessing and model
provenance so incompatible data, encoder, or Inception changes do not silently
reuse old results. All current benchmarks are unconditional; labels remain
dataset metadata and are not passed to the vector field.
