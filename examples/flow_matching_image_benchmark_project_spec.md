# Project Specification: Flow-Matching Image Benchmark Harness

**Status:** implementation specification  
**Scope:** unconditional image generation benchmarks for comparing Flow Matching training/optimization methods  
**Primary framework:** JAX + Flax NNX  
**Reference codebase to reuse where possible:** [willisma/diffuse_nnx](https://github.com/willisma/diffuse_nnx)  
**Reference Flow Matching image code where useful:** [atong01/conditional-flow-matching (TorchCFM)](https://github.com/atong01/conditional-flow-matching)  
**Specification date:** 2026-08-11

---

## 1. Goal

Build a reusable experiment harness for **unconditional Flow Matching image generation** over a hierarchy of datasets and representation/model sizes.

The harness must make it possible to vary independently:

1. dataset;
2. image resolution / preprocessing;
3. representation:
   - raw pixels (`none`);
   - the existing project small autoencoder (`ae`) with configurable latent dimension;
   - the pretrained Stable-Diffusion-style VAE from DiffuseNNX (`vae`);
4. RHS / vector-field architecture:
   - MLP;
   - U-Net variants;
   - SiT/DiT transformer variants;
5. optimizer/training method, using the project's **existing** optimizer/training implementation;
6. evaluation protocol.

The scientific objective is to compare **training methods**, so all non-optimizer components must be configurable but reproducible and held fixed within a comparison.

This project is for **unconditional generation only**. Dataset labels may be retained in metadata but must not be passed to the generative model unless a future extension explicitly enables conditioning.

---

## 2. Non-goals

Do **not** reimplement or redesign:

- the project's existing Flow Matching training-method implementations;
- the project's existing optimizer algorithms;

Instead, integrate them through small registries/adapters and define the configuration contract they must receive.

Do not add text conditioning, classifier-free guidance, CLIP/text encoders, or prompt handling.

Do not train the Stable Diffusion VAE from scratch.

---

## 3. Reference-implementation policy

Prefer adapting/importing tested reference code over rewriting algorithms.

### 3.1 DiffuseNNX

Pin the installable DiffuseNNX fork to Git commit
`da5f2b79497722931d279b012c90bec61050466b` in the project dependency
metadata. Do not depend on an unpinned `main` branch for paper experiments.

The pinned fork uses a standard ``src/diffuse_nnx`` package layout. Integration
tests must import its canonical modules from an installed distribution and must
not clone a source checkout or modify ``PYTHONPATH`` for benchmark runs.

DiffuseNNX currently provides:

- JAX + Flax NNX infrastructure;
- the Stable Diffusion VAE in `networks/encoders/sd_vae.py`;
- RGB/pass-through encoder support;
- DiT/LightningDiT transformer implementations;
- the SiT / continuous Flow Matching interface in `interfaces/continuous.py`;
- deterministic/stochastic samplers;
- an FID pipeline in `eval/fid.py`;
- an NNX InceptionV3 FID feature extractor and pretrained FID weights.

Important current-repository caveat:

> As of 2026-08-11, the DiffuseNNX README describes EDM/EDM2 U-Nets and its roadmap lists an NNX U-Net implementation, but the current `networks/` tree does not contain an importable `unets/` directory. Therefore the implementation must **check the pinned revision** before attempting a DiffuseNNX U-Net import. If no U-Net exists, implement/port the U-Net locally instead of inventing a nonexistent reference import.

Use adapters around DiffuseNNX rather than modifying its source when practical.

### 3.2 TorchCFM

Use TorchCFM as the architectural reference for the standard CIFAR-10 Flow Matching U-Net if DiffuseNNX does not provide a usable U-Net in the pinned revision.

The TorchCFM CIFAR-10 reference uses approximately:

```text
num_res_blocks = 2
base_channels = 128
channel_mult = [1, 2, 2, 2]
num_heads = 4
num_head_channels = 64
attention_resolution = 16
dropout = 0.1
```

Reference:
`torchcfm.models.unet.unet.UNetModelWrapper`.

The local implementation should be Flax NNX and should reproduce the architecture semantics, not necessarily PyTorch parameter naming.

---

## 4. Benchmark hierarchy

Implement the following datasets as first-class registry entries.

| Registry name | Dataset | Purpose | HF dataset | Native / target resolution | Expected split policy | Default representation |
|---|---|---|---|---|---|---|
| `mnist` | MNIST | lowest-cost real-image sanity check | `ylecun/mnist` | 28×28 grayscale | HF train/test; carve validation from train | none or small AE |
| `cifar10` | CIFAR-10 | main small natural-image benchmark | `uoft-cs/cifar10` | 32×32 RGB | HF train/test; carve validation from train | none |
| `flowers102` | Oxford Flowers-102 | small natural-image benchmark with rich texture/colour | `pufanyi/flowers102` | preset 64×64 and 256×256 RGB | use provided train/validation/test | none at 64; VAE or AE at 256 |
| `afhq_cat` | AFHQ Cats | high-quality single-domain natural images | `bitmind/AFHQ` | source 512×512; default target 256×256 | filter cat; recover original train/test from `filename`; carve validation from train | VAE or AE |
| `lsun_church` | LSUN Churches | larger natural-scene benchmark | `tglcourse/lsun_church_train` | default 256×256 RGB | use provided train/test; carve validation from train | VAE or AE |
| `ffhq64` | FFHQ-64 | medium-size face benchmark in pixel space | `Dmini/FFHQ-64x64` | 64×64 RGB | deterministic project split because HF mirror has one split | none |
| `imagenet64` | ImageNet-1K 64×64 | large-scale low-resolution benchmark | `benjamin-paine/imagenet-1k-64x64` | 64×64 RGB | HF train/validation; validation is final real reference | none |
| `imagenet256` | ImageNet-1K | final large-scale latent benchmark | `ILSVRC/imagenet-1k` | preprocess to 256×256 RGB | HF train/validation; validation is final real reference | VAE |

Optional extension after the required milestones:

| Registry name | Dataset | HF dataset | Comment |
|---|---|---|---|
| `tiny_imagenet` | Tiny ImageNet, 200 classes | `zh-plus/tiny-imagenet` | useful intermediate 64×64 / 100k-image benchmark |

### 4.1 Known dataset sizes

Use these as sanity checks, not as hard-coded download logic:

- MNIST: 60,000 train / 10,000 test.
- CIFAR-10: 50,000 train / 10,000 test.
- Oxford Flowers-102: 8,189 total images. The selected HF mirror provides an 80/10/10 split.
- AFHQ full dataset: 15,803 images across cat/dog/wild. For the cat subset, preserve the original train/test distinction encoded in the path when possible.
- LSUN Church HF mirror: 119,915 train / 6,312 test.
- FFHQ-64 HF mirror: 70,000 images in one split.
- ImageNet-1K: 1,281,167 train / 50,000 validation / 100,000 test in the source dataset.

### 4.2 ImageNet access

`ILSVRC/imagenet-1k` is gated/licensed. Dataset preparation must:

- use the Hugging Face token from environment/config;
- fail with a clear access message if the user has not accepted the ImageNet terms;
- never silently substitute another dataset.

For `imagenet64`, prefer the already-resized HF derivative above so the benchmark does not repeatedly resample 1.28M images.

---

## 5. Milestone 1 — Dataset download and preprocessing

### 5.1 Required API

Implement a dataset registry with a common conceptual contract:

```python
DatasetSpec(
    name=...,
    hf_id=...,
    image_key=...,
    label_key=...,
    default_resolution=...,
    channels=...,
    split_recipe=...,
    preprocessing=...,
)
```

Provide functions equivalent to:

```python
download_dataset(spec, cache_dir, hf_token=None) -> DatasetManifest
load_split(manifest, split, batch_size, seed, ...) -> iterator
```

Exact names may follow the existing project style.

### 5.2 Use Hugging Face as the download layer

Use `datasets.load_dataset` and/or `huggingface_hub` rather than dataset-specific web scrapers.

Datasets that ship as a bare image archive (e.g. `Dmini/FFHQ-64x64`, a single
`ffhq-64x64.zip`) need a `zip_imagefolder` loader: the Hub's automatic parquet
conversion of such repos can declare an incompatible `ClassLabel` feature that
the default `datasets` loader rejects. Set `loader="zip_imagefolder"` and
`archive_file=<name>` on the `DatasetSpec`; the loader downloads the archive,
extracts it once under `cache_dir/raw/<name>`, and loads it as a local
`imagefolder`.

Requirements:

- support `cache_dir`;
- support offline reuse after download;
- record dataset repository revision/hash when available;
- write a local manifest containing:
  - HF dataset ID;
  - HF revision;
  - split names and counts;
  - preprocessing resolution;
  - split seed where project-generated splits are used;
  - image normalization convention.

### 5.3 Deterministic split policy

For datasets without a suitable validation split:

- use a fixed project-wide split seed;
- derive split membership from stable example indices or stable filenames;
- write split indices to disk;
- never regenerate different train/val/test assignments between optimizer runs.

Recommended defaults:

- MNIST: reserve 5,000 examples from original training set for validation.
- CIFAR-10: reserve 5,000 examples from original training set for validation.
- AFHQ-cat: preserve source `train/cat/...` vs `test/cat/...`; reserve 10% of original cat train as validation.
- LSUN Church: preserve HF test; reserve 5,000 examples from train for validation.
- FFHQ-64: deterministic 60k train / 5k validation / 5k test split.
- ImageNet: official validation is final evaluation; optionally reserve a deterministic subset from training for cheap per-epoch FM validation if desired.

### 5.4 AFHQ filtering

For `bitmind/AFHQ`, filter by `filename`:

```text
train/cat/*
test/cat/*
```

Do not use labels from dog/wild images.

### 5.5 Preprocessing

Internal tensor convention should be consistent with JAX/NNX and DiffuseNNX:

```text
NHWC
```

Raw model-space image convention:

```text
float32 or configured compute dtype
[-1, 1]
```

Evaluation image convention:

```text
uint8
[0, 255]
NHWC
```

Training transforms:

- MNIST:
  - no horizontal flip;
  - native 28×28 unless config overrides;
  - one channel for pixel/AE experiments;
  - only replicate to RGB inside an evaluation adapter if the metric requires RGB.
- CIFAR-10:
  - native 32×32;
  - optional horizontal flip controlled by config.
- Natural image datasets:
  - deterministic square crop + resize definition must be shared by all optimizer runs;
  - training may use horizontal flip `p=0.5`;
  - evaluation must be deterministic;
  - avoid aggressive `RandomResizedCrop` by default because it changes the target distribution.

For ImageNet-256 use the same center-crop/resize convention across all methods.

### 5.6 Download acceptance criteria

A dataset-download smoke test must:

1. download or resolve each dataset;
2. load at least one batch from every required split;
3. verify expected channel count and configured resolution;
4. verify deterministic split membership across two independent processes;
5. print/write a manifest summary.

Do not require full ImageNet download in ordinary CI; use mocked metadata or a tiny fixture for CI and a separate integration test.

---

## 6. Milestone 2 — Encoder abstraction

Implement three encoder modes with one common adapter contract.

Conceptual contract:

```python
encode(images, rng=None) -> latent
decode(latent, rng=None) -> images
latent_shape(input_shape) -> tuple
is_stochastic: bool
```

For stochastic encoders, optionally expose:

```python
encode_stats(images) -> (mean, std)
sample_from_stats(mean, std, rng) -> latent
```

### 6.1 Encoder `none`

Identity/pass-through representation.

Behavior:

```text
input image -> normalized pixel tensor -> Flow Matching state
```

No learned parameters.

Use DiffuseNNX's RGB/pass-through encoder semantics where useful, but a trivial local adapter is acceptable.

### 6.2 Encoder `ae`

Use the project's existing small autoencoder implementation.

The benchmark harness must only add an adapter and experiment integration.

Required configuration:

```yaml
encoder:
  type: ae
  latent_dim: 64
  checkpoint: /path/to/checkpoint
  train_if_missing: true
  frozen_during_flow_training: true
```

Requirements:

- latent dimension is variable;
- support loading a previously trained AE;
- if the existing AE training entry point is invoked, save its checkpoint separately from the Flow Matching checkpoint;
- freeze the AE during Flow Matching experiments;
- expose reconstruction metrics on the held-out split;
- record AE checkpoint hash in every experiment result.

If the AE produces a vector latent, MLP is the default RHS.  
If it produces a spatial latent, U-Net and SiT may be used.

Do not silently reshape arbitrary vector latents into a fake image grid for U-Net/SiT. Require an explicit configured reshape if such an experiment is desired.

### 6.3 Encoder `vae`

Reuse DiffuseNNX `networks/encoders/sd_vae.py::StabilityVAE`.

The reference VAE has:

```text
latent_channels = 4
spatial downsample factor = 8
scale factor = 0.18215
block widths = 128, 256, 512, 512
layers/block = 2
```

For 256×256 images:

```text
NHWC RGB: 256×256×3
    ->
VAE latent: 32×32×4
```

#### VAE checkpoint

Use the checkpoint path expected by DiffuseNNX:

```text
vae_trial1.pkl
```

DiffuseNNX currently downloads it automatically if absent from:

```python
utils.download_blob(
    "will-data",
    "stats/vae_trial1.pkl",
    ckpt_path,
)
```

At the pinned revision the object is not publicly readable without Google Cloud
credentials, and upstream issue #3 states that public checkpoint release is
still pending. Therefore asset preparation may require application-default GCP
credentials. A VAE run must also provide a trusted SHA-256 obtained independently
from the download; the pickle checkpoint must never be loaded using only a
trust-on-first-use checksum produced by the same download operation.

The project must provide an explicit asset-preparation function/command that instantiates or calls the same download helper so the checkpoint can be prepared **before** a training job starts.

Required behavior:

- configurable local checkpoint destination;
- idempotent download;
- clear error if download fails;
- log file size and checksum after successful download;
- allow `auto_download: false` for fully offline runs;
- never train this VAE as part of the benchmark.
- require `expected_sha256` before deserializing the pickle checkpoint.

Example config:

```yaml
encoder:
  type: vae
  implementation: diffuse_nnx
  checkpoint: artifacts/encoders/vae_trial1.pkl
  expected_sha256: <trusted 64-character checksum>
  auto_download: true
  sample_posterior: true
  cache_latents: true
```

#### VAE latent caching

Strongly prefer precomputing encoder posterior statistics for expensive datasets.

Cache:

```text
mean
std
stable sample identifier
```

During Flow Matching training sample

```text
z = mean + std * epsilon
```

with the experiment RNG.

This avoids repeatedly executing the frozen VAE and makes optimizer wall-clock comparisons cleaner.

The cache must be tied to:

- dataset revision;
- split;
- preprocessing resolution/crop;
- VAE checkpoint checksum;
- VAE latent scaling convention.

If any of those change, invalidate the cache.

#### VAE compatibility policy

Recommended:

- 32×32 CIFAR: do not use SD-VAE by default; use pixels or the small AE.
- 64×64 images: VAE is allowed experimentally but is not the default because the resulting spatial latent is only 8×8×4.
- 256×256 images: VAE is the default latent representation.
- 512×512: supported, but benchmark cost is higher; AFHQ default benchmark target should be 256 unless explicitly testing 512.

---

## 7. Milestone 3 — RHS / vector-field architectures

All RHS architectures must expose one project-level interface such as:

```python
v = rhs(x_t, t)
```

for unconditional generation.

If a reused reference network returns `(prediction, features)`, the adapter must expose only the prediction to the existing trainer while optionally retaining features for diagnostics.

Output shape must equal input state shape.

Time is always required.

No class/text conditioning is required.

### 7.1 MLP

Implement a generic Flax NNX MLP for vector states.

Default use cases:

- raw MNIST pixels after flattening;
- small-AE vector latents;
- controlled low-dimensional image-latent experiments.

Required configurable fields:

```yaml
rhs:
  type: mlp
  hidden_dims: [512, 512, 512]
  activation: silu
  time_embedding:
    type: sinusoidal
    dim: 128
  residual: false
```

Allow configurable residual blocks, but keep a simple plain MLP as the baseline.

The architecture must concatenate or otherwise inject a time embedding at a clearly defined point and map back to exactly the input vector dimension.

### 7.2 U-Net

Primary architecture for pixel-space image Flow Matching.

Implement variants through one configurable NNX U-Net, not multiple unrelated classes.

Recommended presets:

```yaml
unet_small:
  base_channels: 64
  channel_mult: [1, 2, 2, 2]
  num_res_blocks: 2
  attention_resolutions: [16]
  dropout: 0.0
```

```yaml
unet_cifar_reference:
  base_channels: 128
  channel_mult: [1, 2, 2, 2]
  num_res_blocks: 2
  num_heads: 4
  num_head_channels: 64
  attention_resolutions: [16]
  dropout: 0.1
```

```yaml
unet_large:
  base_channels: 192
  channel_mult: [1, 2, 3, 4]
  num_res_blocks: 2
  attention_resolutions: [16, 8]
  dropout: 0.1
```

Requirements:

- sinusoidal/Fourier time embedding followed by an MLP;
- residual blocks with time conditioning;
- skip connections;
- attention at configurable spatial resolutions;
- input/output channel count derived from representation;
- no class embedding in unconditional mode;
- parameter-count utility for every instantiated model.

Reference priority:

1. if the pinned DiffuseNNX revision actually has a tested NNX U-Net, use/adapt it;
2. otherwise port the TorchCFM CIFAR U-Net semantics to NNX;
3. add a parity/shape test against the reference architecture configuration.

### 7.3 SiT / DiT

For this project, “SiT RHS” means:

> Use the DiffuseNNX DiT-style transformer backbone as the vector-field network under the SiT / Flow Matching interface, with all class conditioning disabled.

Reuse DiffuseNNX rather than writing a transformer from scratch.

DiffuseNNX currently defines DiT size presets approximately as:

| Variant | hidden size | depth | heads |
|---|---:|---:|---:|
| S | 384 | 12 | 6 |
| B | 768 | 12 | 12 |
| L | 1024 | 24 | 16 |
| XL | 1152 | 28 | 16 |

Required project presets:

```text
sit_s_2
sit_b_2
```

where `/2` indicates patch size 2 in the spatial state.

L/XL should be supported if the reference code makes them trivial, but they are not required for the first benchmark milestone.

Configuration example:

```yaml
rhs:
  type: sit
  implementation: diffuse_nnx
  variant: B
  patch_size: 2
  class_conditioning: false
```

The adapter must ensure that no label embedding affects the network. If the DiffuseNNX DiT implementation requires a `num_classes` argument structurally, implement a clean unconditional mode or a constant null token and test that labels are not read from the dataset.

Do not use classifier-free guidance in unconditional experiments.

### 7.4 Architecture compatibility validation

Validate configuration before training:

- MLP requires a vector state unless explicit flatten/unflatten adapter is enabled.
- U-Net requires a spatial state `(H, W, C)`.
- SiT requires a spatial state whose dimensions are divisible by patch size.
- RHS output shape must equal encoder state shape.
- VAE + 256×256 implies 32×32×4 and is compatible with U-Net/SiT.
- `none` + CIFAR implies 32×32×3 and is compatible with U-Net/SiT.
- AE compatibility depends on its declared latent shape.

---

## 8. Milestone 4 — Evaluation

Evaluation must separate:

1. optimization quality;
2. generated-sample quality;
3. representation quality when an encoder is used.

### 8.1 Fixed held-out Flow Matching loss

This is required for **every** dataset and architecture.

Evaluate the existing project's Flow Matching objective on a fixed held-out set using:

- fixed sample IDs;
- fixed evaluation RNG seed;
- fixed `t` and noise sampling protocol;
- no training augmentation;
- no parameter updates.

Log:

```text
val_fm_loss
```

This is the primary metric for answering:

> Which optimizer solves the same training problem more effectively?

The metric implementation should call the existing training-method loss implementation rather than reimplement the mathematical objective.

### 8.2 FID

Reuse DiffuseNNX's FID pipeline:

- `eval/fid.py`;
- `eval/inception.py`;
- `eval/inception_v3_weights_fid.pickle`;
- `eval.utils` detector/download helpers.

The implementation should wrap these functions rather than fork them.

Required capabilities:

- compute real Inception statistics once per dataset/preprocessing configuration;
- cache `mu` and `sigma`;
- generate samples from a checkpoint;
- decode through AE/VAE if needed;
- convert to uint8 `[0,255]`;
- compute FID;
- deterministic evaluation seed;
- configurable generated sample count.

Recommended sample counts:

- smoke/validation evaluation: 1k–10k generated samples;
- final CIFAR/ImageNet-style evaluation: 50k generated samples;
- small datasets: still allow 50k generated samples, but use all available held-out real examples and report real-example count.

Cache key for real FID statistics must include:

```text
dataset revision
evaluation split
resolution
crop/resize policy
channel conversion
```

### 8.3 KID

Implement KID as an additional metric for small datasets because FID is less reliable with small real sample counts.

Required by default for:

- Flowers-102;
- AFHQ-cat;
- FFHQ project test split if only 5k real examples are used;
- MNIST if using the generic Inception feature pipeline.

Reuse the same Inception features produced by DiffuseNNX. Only implement the polynomial-kernel MMD/KID aggregation locally.

Report mean and standard error / standard deviation across KID subsets.

### 8.4 MNIST note

Inception-FID is not an especially natural metric for 28×28 grayscale digits.

For consistency, the generic image metric adapter may:

1. replicate grayscale to 3 channels;
2. resize only inside the feature extractor path as required by the FID implementation.

But the report must mark this as generic Inception-FID and not imply it is directly comparable to papers using a digit-specific feature network.

For MNIST, `val_fm_loss` is the primary optimizer metric; generated grids and optional KID/FID are secondary.

### 8.5 Encoder reconstruction metrics

For learned `ae`:

- reconstruction MSE;
- PSNR;
- optionally LPIPS if already available in the project.

For frozen DiffuseNNX VAE, reconstruction metrics are diagnostic only and need not be recomputed every experiment if the dataset/preprocessing/checkpoint tuple is unchanged.

### 8.6 Evaluation result schema

Each checkpoint evaluation should produce structured output comparable to:

```json
{
  "step": 100000,
  "epoch": 120.0,
  "wall_clock_train_s": 12345.6,
  "val_fm_loss": 0.0123,
  "fid": 8.7,
  "fid_num_fake": 50000,
  "fid_num_real": 10000,
  "kid_mean": 0.0041,
  "kid_std": 0.0003,
  "encoder_recon_mse": null
}
```

Metrics that are not applicable may be null/omitted.

---

## 9. Milestone 5 — Optimizer/training-method integration

Do **not** implement the training methods in this milestone; use the existing project implementation.

The benchmark harness only needs a configuration/registry integration layer.

Required config contract:

```yaml
optimizer:
  name: adamw
  kwargs:
    learning_rate: 1.0e-4
    beta1: 0.9
    beta2: 0.999
    eps: 1.0e-8
    weight_decay: 0.0
```

Custom existing methods must be selectable by registry name with arbitrary method-specific kwargs:

```yaml
optimizer:
  name: <existing_custom_method>
  kwargs:
    ...
```

Required baseline availability:

- Adam;
- AdamW;
- the project's existing custom Flow Matching training/optimization methods.

The benchmark layer must not bake optimizer-specific logic into dataset, encoder, RHS, or evaluator code.

### 9.1 Fair-comparison logging

Because second-order/LM-style methods can have very different cost per update, every run must log at least:

- optimizer step;
- effective epoch;
- examples seen;
- wall-clock training time;
- wall-clock evaluation time;
- parameter count;
- peak accelerator memory if available;
- number of forward evaluations if the training implementation exposes it;
- number of gradient/VJP/JVP evaluations if the training implementation exposes it.

Do not estimate these counts if the training method cannot report them; record `null`.

The benchmark must support plotting:

```text
val FM loss vs epoch
val FM loss vs wall-clock
FID/KID vs epoch
FID/KID vs wall-clock
```

and, when available:

```text
quality vs forward/backward/JVP/VJP-equivalent compute
```

---

## 10. Milestone 6 — Config-driven experiment launch

Reuse the project's existing CLI implementation.

The new work is to define and validate the experiment config schema and wire the new registries into the existing launcher.

### 10.1 Required top-level config concept

At minimum:

```yaml
experiment:
  name: cifar10_unet_adamw
  seed: 0
  output_dir: runs/cifar10_unet_adamw

problem:
  dataset:
    name: cifar10
    resolution: 32
    cache_dir: data/hf
  encoder:
    type: none

rhs:
  type: unet
  variant: cifar_reference

optimizer:
  name: adamw
  kwargs:
    learning_rate: 1.0e-4

evaluation:
  val_fm_loss: true
  fid:
    enabled: true
    num_samples_final: 50000
  kid:
    enabled: false
```

The existing training-method config may be included in the project's existing format and is intentionally not specified here.

### 10.2 Encoder examples

No encoder:

```yaml
problem:
  dataset:
    name: cifar10
    resolution: 32
  encoder:
    type: none
```

Small project AE:

```yaml
problem:
  dataset:
    name: mnist
    resolution: 28
  encoder:
    type: ae
    latent_dim: 64
    checkpoint: artifacts/ae/mnist_latent64.ckpt
    train_if_missing: true
```

DiffuseNNX VAE:

```yaml
problem:
  dataset:
    name: imagenet256
    resolution: 256
  encoder:
    type: vae
    implementation: diffuse_nnx
    checkpoint: artifacts/encoders/vae_trial1.pkl
    auto_download: true
    sample_posterior: true
    cache_latents: true
```

### 10.3 Config validation

Fail before accelerator initialization for:

- unknown dataset;
- unavailable HF credentials for gated dataset;
- missing encoder checkpoint with `auto_download/train_if_missing=false`;
- incompatible encoder/RHS state shape;
- patch size not dividing SiT input;
- requested resolution not supported by a strict dataset preset;
- evaluation requested without a decoder for latent models;
- invalid split name.

Print the fully resolved config and save it to the run directory before training.

### 10.4 Canonical JSON and run expansion

JSON is the canonical configuration format for this implementation. YAML
snippets elsewhere in this specification are illustrative representations of
the same schema.

Only the top-level `methods` array and each method's `n_restarts` field expand
into planned runs. Lists inside dataset, encoder, RHS, optimizer kwargs,
evaluation, or resource configuration are literal values. In particular,
fields such as `hidden_dims`, `channel_mult`, and `attention_resolutions` must
never be interpreted as sweep axes.

To compare two hyperparameter settings for the same optimizer, include two
entries in `methods`, each with explicit scalar kwargs. This avoids ambiguous
list semantics and makes every planned run independently serializable.

### 10.5 Resource scheduling and data parallelism

The runner resource contract is:

```json
{
  "resources": {
    "gpu_ids": [0, 1, 2, 3],
    "gpus_per_run": 2,
    "max_concurrent_runs": 2,
    "data_parallel": true
  }
}
```

- `gpu_ids` is the scheduler-visible accelerator pool.
- `gpus_per_run` is the number of accelerators reserved for one planned run.
- `max_concurrent_runs` is an upper bound on independently executing run
  processes.
- Effective concurrency is additionally bounded by
  `floor(len(gpu_ids) / gpus_per_run)`.
- A worker receives its reserved devices through `CUDA_VISIBLE_DEVICES` before
  importing JAX.
- When `gpus_per_run > 1`, training uses replicated data parallelism: model and
  optimizer state are replicated and the global batch is sharded over a named
  data mesh. Losses, gradients, Flow Matching pullback products, and reported
  metrics must represent the global batch.

Replicated data-parallel behavior is required for Adam/AdamW, NGD, and Anderson.
Single-device and multi-device updates must be checked for numerical parity on
a deterministic small problem. Global batch size must be divisible by the
number of devices unless an explicitly documented padding/masking policy is
enabled.

Concurrent runs use independent spawned processes rather than optimizer-lane
`vmap`. Run seeds and named RNG streams are derived from the resolved run
identity, not worker assignment or scheduling order.

The initial distributed implementation is single-host: one worker process may
span several locally visible accelerators. Multi-host JAX meshes must fail
explicitly rather than silently launching independent replicas.

### 10.6 Checkpoint and resume contract

Each checkpoint must contain enough state to resume training, including model
parameters, optimizer/method state, step, examples seen, wall-clock accumulator,
and named RNG state. Checkpoint publication must be atomic or versioned so an
interrupted write cannot destroy the last valid checkpoint.

Metrics are appended to JSONL during training. Compact NumPy/CSV summaries may
be generated after a run, but must not be the only durable metric record.

---

## 11. Recommended benchmark presets

Provide ready-to-run config files for at least the following.

### Tier 1 — cheap

```text
MNIST / none / MLP
MNIST / AE-32 / MLP
MNIST / AE-64 / MLP
```

Purpose: optimizer debugging and latent-dimension scaling.

### Tier 2 — main small-image benchmark

```text
CIFAR-10 / none / UNet-small
CIFAR-10 / none / UNet-CIFAR-reference
```

Purpose: primary recognized image-generation benchmark without VAE confounding.

### Tier 3 — first natural-image latent benchmark

```text
Flowers-102 64 / none / UNet-small
Flowers-102 256 / VAE / latent-UNet
Flowers-102 256 / VAE / SiT-S/2
```

Purpose: compare pixel vs latent representations on a small but visually rich dataset.

### Tier 4 — medium natural-image benchmarks

```text
FFHQ-64 / none / UNet-CIFAR-reference
AFHQ-cat 256 / VAE / latent-UNet
LSUN-Church 256 / VAE / latent-UNet
LSUN-Church 256 / VAE / SiT-S/2
```

### Tier 5 — scaling

```text
ImageNet-64 / none / UNet
ImageNet-64 / none / SiT-S/2
ImageNet-256 / VAE / SiT-S/2
ImageNet-256 / VAE / SiT-B/2
```

ImageNet class labels must be ignored for unconditional runs.

---

## 12. Suggested project structure

Adapt to the existing repository rather than forcing these exact directories.

```text
project/
├── examples/
│   ├── flow_matching_image_benchmark_runner.py
│   └── image_benchmarks/
│       ├── datasets/
│       │   ├── registry.py
│       │   ├── hf_loader.py
│       │   ├── splits.py
│       │   └── transforms.py
│       ├── encoders/
│       │   ├── base.py
│       │   ├── identity.py
│       │   ├── project_ae.py
│       │   └── diffuse_vae.py
│       ├── rhs/
│       │   ├── registry.py
│       │   ├── mlp.py
│       │   ├── unet.py
│       │   └── diffuse_sit.py
│       ├── evaluation/
│       │   ├── validation.py
│       │   ├── fid.py
│       │   ├── kid.py
│       │   └── reconstruction.py
│       ├── assets/
│       ├── training/
│       ├── config.py
│       └── configs/
│           ├── mnist_mlp.json
│           ├── cifar10_unet.json
│           ├── flowers256_vae_unet.json
│           ├── flowers256_vae_sit_s2.json
│           ├── afhqcat256_vae_unet.json
│           ├── lsun256_vae_unet.json
│           ├── ffhq64_unet.json
│           ├── imagenet64_unet.json
│           └── imagenet256_vae_sit_b2.json
└── tests/
    └── benchmarks/
```

---

## 13. Reproducibility requirements

Every run directory must store:

- resolved config;
- git commit of this project;
- pinned DiffuseNNX commit;
- dataset HF ID and revision;
- deterministic split manifest/hash;
- encoder checkpoint path and checksum;
- RHS parameter count;
- optimizer/training-method name and serialized kwargs;
- seed;
- package/environment snapshot;
- accelerator/device metadata;
- checkpoints;
- metric JSON/JSONL/CSV output;
- generated sample grids;
- final evaluation summary.

Randomness must be separated into named RNG streams where practical:

```text
dataset_shuffle
augmentation
encoder_sampling
fm_noise
fm_time
model_dropout
sampling
evaluation
```

Optimizer comparisons must use the same model initialization seed and, where the existing trainer permits, the same dataset order / FM sampling RNG protocol.

---

## 14. Tests

### 14.1 Unit tests

Dataset:

- registry lookup;
- split determinism;
- image normalization range;
- target shape.

Encoder:

- identity round trip;
- AE adapter shape;
- VAE checkpoint download mocked;
- VAE encode/decode shape;
- VAE latent scale behavior;
- latent cache key invalidation.

RHS:

- MLP output shape;
- U-Net output shape at 28/32/64 and latent 32×32×4;
- SiT output shape;
- unconditional SiT does not require labels;
- parameter count > 0 and stable for a fixed config.

Evaluation:

- FID wrapper matches DiffuseNNX FID on the same fixed feature statistics;
- KID returns approximately zero when comparing a feature set with itself;
- latent sample evaluation always calls the correct decoder;
- FM validation uses fixed RNG.

Config:

- valid example configs resolve;
- invalid encoder/RHS pair fails;
- missing gated dataset access produces actionable error.

### 14.2 Integration tests

Required small integration jobs:

1. MNIST + none + tiny MLP for a few steps;
2. CIFAR-10 + none + tiny U-Net for a few steps;
3. Flowers-102 + VAE + tiny U-Net for a few steps;
4. Flowers-102 + VAE + SiT-S/2 for a few steps;
5. generate a small sample batch and run FID feature extraction.

Full ImageNet is excluded from normal CI.

---

## 15. Milestone acceptance criteria

### M1 — datasets

Complete when all required dataset registry entries can be downloaded/resolved from Hugging Face, split deterministically, transformed to configured resolution, and iterated through one common loader interface.

### M2 — encoders

Complete when `none`, project `ae`, and DiffuseNNX `vae` satisfy the same adapter contract; the VAE checkpoint downloads automatically; and latent caching works.

### M3 — RHS

Complete when MLP, at least two U-Net presets, and DiffuseNNX SiT-S/2 + SiT-B/2 can all be instantiated from config and produce an output with the same shape as their input state.

### M4 — evaluation

Complete when fixed held-out FM loss and FID work end-to-end for pixel and latent models, KID works on cached Inception features, and real FID statistics are cacheable.

### M5 — optimizer integration

Complete when the existing optimizer/training-method registry can be selected solely through config and the benchmark layer contains no method-specific training logic.

### M6 — config launch

Complete when each recommended benchmark preset can be launched through the existing CLI from one config file, with preflight validation and reproducibility metadata.

---

## 16. Implementation order

Implement in this order to keep debugging local:

1. Dataset registry + MNIST/CIFAR.
2. `none` encoder.
3. MLP.
4. fixed validation FM-loss wrapper.
5. U-Net small + CIFAR reference.
6. DiffuseNNX FID adapter.
7. Flowers/AFHQ/LSUN/FFHQ dataset adapters.
8. project AE adapter.
9. DiffuseNNX VAE adapter + checkpoint download.
10. latent caching.
11. latent U-Net.
12. DiffuseNNX SiT adapter.
13. KID.
14. ImageNet-64.
15. gated ImageNet-256.
16. benchmark config suite and integration tests.

Do not begin with ImageNet or SiT. The first complete end-to-end target should be:

```text
CIFAR-10
+ raw pixels
+ U-Net
+ existing Flow Matching trainer
+ existing optimizer
+ val FM loss
+ FID
```

The first latent target should be:

```text
Flowers-102 @ 256
+ frozen DiffuseNNX VAE
+ latent U-Net
+ existing Flow Matching trainer
+ val FM loss
+ FID/KID
```

---

## 17. Notes on Flow Matching convention

The benchmark harness should not independently define the training method; it should call the existing project implementation.

For compatibility with DiffuseNNX, note that its current SiT/Flow Matching interface uses the convention

```text
x_t = (1 - t) * x_data + t * noise
target = noise - x_data
```

and the network predicts the tangent.

If the project's existing trainer uses the opposite time orientation,

```text
x_t = (1 - t) * noise + t * x_data
target = x_data - noise
```

that is mathematically equivalent after reversing time, but **do not mix conventions inside one experiment**.

Create an adapter boundary and test the sign/time convention explicitly when reusing DiffuseNNX network/interface code.

---

## 18. Sources / reference links

Primary references used for this specification:

- DiffuseNNX repository:  
  https://github.com/willisma/diffuse_nnx

- DiffuseNNX SD-VAE implementation:  
  https://raw.githubusercontent.com/willisma/diffuse_nnx/main/networks/encoders/sd_vae.py

- DiffuseNNX Flow Matching / SiT interface:  
  https://raw.githubusercontent.com/willisma/diffuse_nnx/main/interfaces/continuous.py

- DiffuseNNX FID implementation:  
  https://raw.githubusercontent.com/willisma/diffuse_nnx/main/eval/fid.py

- DiffuseNNX ImageNet / DiT presets:  
  https://raw.githubusercontent.com/willisma/diffuse_nnx/main/configs/common_specs.py

- DiffuseNNX DiT/ImageNet config:  
  https://raw.githubusercontent.com/willisma/diffuse_nnx/main/configs/dit_imagenet.py

- TorchCFM repository:  
  https://github.com/atong01/conditional-flow-matching

- TorchCFM CIFAR-10 FID/configuration example:  
  https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/compute_fid.py

Hugging Face datasets:

- MNIST: https://huggingface.co/datasets/ylecun/mnist
- CIFAR-10: https://huggingface.co/datasets/uoft-cs/cifar10
- Flowers-102: https://huggingface.co/datasets/pufanyi/flowers102
- AFHQ: https://huggingface.co/datasets/bitmind/AFHQ
- LSUN Church: https://huggingface.co/datasets/tglcourse/lsun_church_train
- FFHQ-64: https://huggingface.co/datasets/Dmini/FFHQ-64x64
- ImageNet-1K 64×64: https://huggingface.co/datasets/benjamin-paine/imagenet-1k-64x64
- ImageNet-1K: https://huggingface.co/datasets/ILSVRC/imagenet-1k
- Optional Tiny ImageNet: https://huggingface.co/datasets/zh-plus/tiny-imagenet

---

## 19. Final deliverable

The completed project should allow a new benchmark to be defined primarily by configuration, e.g.:

```yaml
problem:
  dataset:
    name: flowers102
    resolution: 256
  encoder:
    type: vae
    checkpoint: artifacts/encoders/vae_trial1.pkl

rhs:
  type: unet
  variant: cifar_reference

optimizer:
  name: <existing_optimizer>
  kwargs: {...}

evaluation:
  val_fm_loss: true
  fid:
    enabled: true
  kid:
    enabled: true
```

Changing the optimizer must not require modifying dataset, encoder, RHS, sampling, or metric code.

Changing `none -> ae -> vae` must not require modifying the existing Flow Matching trainer.

Changing `mlp -> unet -> sit` must not require modifying dataset or optimizer code.

That modularity is the principal architectural requirement of this benchmark harness.
