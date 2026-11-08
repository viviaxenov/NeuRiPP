# Bayesian PDE inverse examples

These implement `jax_bayesian_pde_combined_spec.md`. Default numerical targets
match that specification; reduced grids are explicitly different discretizations.
The generator still starts from an independent standard normal latent. Its
output is the physical parameter field, not a whitened prior coordinate.

## Installation

From the repository root, in an otherwise compatible Python 3.12 environment:

```bash
python -m pip install -e '.[ks]' pandas tqdm
# Both solvers (also suitable for a clean environment):
python -m pip install -e '.[acoustic,ks]' pandas tqdm
python -m pip check
```

The acoustic extra uses immutable j-Wave and JAXDF revisions recorded in
`pyproject.toml`, JAX/JAXlib 0.11.0, Equinox 0.13.7, and Flax 0.12.8.
JAXDF is pinned to `viviaxenov/jaxdf@fbd42a28f2842339a71ee5b4a62dbb580f656402`,
a metadata-only fork allowing JAX >=0.9,<0.12. Upstream main's <0.10 bound
otherwise prevents normal dependency resolution with this project's JAX.
The PyPI j-Wave release uses older JAXDF APIs/dependency bounds. The pinned
upstream API takes `TimeWavePropagationSettings(checkpoint=...)` rather than
the tutorial's direct `checkpoint` keyword; the checkpoint semantics are the
same solver-level choice. KS uses Exponax 0.2.1 and was tested with JAX 0.10.2
and 0.11.0.

Both extras now install normally in conda `neuripp`, without downgrading JAX
or conflicting with `diffuse-nnx`. No `--no-deps` override is required.
If upgrading an environment with another JAXDF 0.3.0 revision already installed,
pip may retain it because the version number is unchanged. Remove that copy
before installing the extras:

```bash
python -m pip uninstall -y jaxdf
python -m pip install -e '.[acoustic,ks]' pandas tqdm
```

For an A100, install the matching CUDA extra, e.g.
`pip install -e '.[acoustic,ks,cuda12]' pandas tqdm`. GPU execution is not yet
verified here.

The fork installation passed `pip check` and all 19 focused tests plus 15
subtests in an isolated JAX 0.11 environment. In `neuripp`, all nine target
tests and the acoustic Adam/NGD parameter-update checks passed; JAX and JAXlib
remained 0.11.0. Its only `pip check` findings, present before installation,
are missing `truststore` for conda and missing `mkl` for mkl-random.

## Getters

Import from `examples.logpdf_targets` (repository root) or `logpdf_targets`
(examples directory). Optional PDE libraries are loaded only on getter calls.

```python
from examples.logpdf_targets import (
    get_logpdf_acoustic_inverse_problem,
    get_logpdf_ks_inverse_problem,
)

target = get_logpdf_acoustic_inverse_problem(
    n_grid=64, num_sources=4, num_receivers=8,
    truth_model="jwave_fwi_three_circles", observation_seed=43,
    source_batch_size=1, checkpoint_strategy="solver_and_chunk",
    data_source_batch_size=None, return_all=True,
)
logpdf = target.logpdf  # construct once; reuse during every training step
ks_logpdf = get_logpdf_ks_inverse_problem(
    num_points=200, num_sensors=20, truth_seed=42, observation_seed=43,
)
```

Callables accept `(..., dim)` flat physical fields, where acoustic dimension is
`n_grid**2` and KS dimension is `num_points`. Densities return the leading batch
shape. `return_all=True` returns `InverseProblem` with `logpdf`, `logprior`,
`loglikelihood`, `forward`, `prior_apply`, `prior_inverse`, `truth`, `clean`,
`noisy`, `noise_std`, and `settings`. Truth is flat; observations have native
shape `(source, time, receiver)` or `(4, sensor)`. Acoustic `truth` is flat
log-contrast `x`; `settings["sound_speed_truth"]` contains the physical phantom.
KS truth is the seeded prior draw. Additive density constants are omitted.
Batched PDE evaluations use sequential `lax.map`.

The prior uses the same positive RMS-normalized FFT multiplier for sampling
and inversion. The zero-frequency mode is retained. The acoustic three-circle
truth is fixed and seed-independent; KS truth and observation seeds are
independent of model/training RNG seeds. A getter creates one fixed dataset,
never new noise inside a loss evaluation. Legacy acoustic run metadata with a
`truth_seed` and no `truth_model` is interpreted as the old Gaussian-truth
version so it is not silently relabeled as the circle phantom.

### Resolution

Acoustic defaults: 256 square grid, 64 sources and receivers, 25.6 mm domain,
reference speed 1630 m/s, CFL 0.25, duration 25 microseconds. Resolution changes
preserve the physical extent and prior length scale, scaling circle radii and
PML width (rounded, minimum one cell). Acquisition points are integer grid
positions; duplicates and points in the PML are rejected. The U-Net requires
grid sizes divisible by four. Source/receiver counts are independently tunable.
The 1 MHz pulse is unchanged: coarse-grid tests verify software, not equivalent
wave accuracy. The acoustic truth is the deterministic three-inclusion
sound-speed phantom from the [j-Wave FWI tutorial](https://ucl-bug.github.io/jwave/notebooks/time_varying/FWI.html): radii 16/20/40, centers
(100,100)/(165,128)/(130,130), and contrasts 5%/2%/3% at the 256-grid
reference resolution. The physical truth is mapped back to the bounded
log-contrast parameterization before synthetic data generation. The Gaussian
prior remains the inference prior; it is no longer used to draw the synthetic
truth. The prior remains defined on the entire grid, including outside the
reconstruction mask.

KS defaults: 200 points on [0,100), 20 sensors, dt=0.1, ETD order 2, observations
at times 5/10/15/20, noise SD 0.1. Sensor indices are
`arange(num_sensors) * num_points // num_sensors`. Changing point count preserves
physical extent, timestep, prior scales, and observation times.

### Source batching and checkpointing

`source_batch_size=1` is sequential, intermediate sizes use parallel chunks,
and a size at least the source count evaluates all sources together. A final
partial chunk is evaluated exactly once. `data_source_batch_size=None` inherits
the inference size; it can be larger for forward-only data generation.

| Strategy | Solver checkpoint | Outer chunk checkpoint |
|---|---|---|
| `none` | off | off |
| `solver` | on | off |
| `chunk` | off | on |
| `solver_and_chunk` | on | on |

Outer rematerialization includes the scalar chunk likelihood reduction. All
sources contribute, without source subsampling or observation-count averaging.
These are execution choices, not changes to the target. Floating-point
reduction/FFT ordering can produce small differences between batch sizes.

Geometry, partitioning, checkpoint wrappers, and JIT wrappers are constructed
once in the getter. Same-shaped/dtyped calls with different field values reuse
compilation. New shapes, dtypes, getter instances, or transformation contexts
can require compilation. The runner caches targets per worker using resolved
settings including seeds and execution choices.

For the laptop use size 1. For a 40-GB A100 start with size 4 or 8, then profile
complete training steps before increasing parallelism. Forward-only memory is
not a reliable bound for gradient or NGD memory.

## Networks

`rhs_acoustic.AcousticUNetRHS`: **509,153** trainable parameters. Native NNX,
16/32/64 widths, time-conditioned residual blocks, eight-group normalization,
two bottleneck blocks. Decoder first fuses the deepest skip at the same
resolution (128->64), then upsamples for 96->32 and 48->16 blocks. This resolves
the specification's three decoder blocks versus two downsamplings.

`rhs_ks.KSDilatedResNetRHS`: **111,073** trainable parameters. Native NNX with
periodic padding, five width-32 blocks, dilations (1,2,4,8,4,2,1), SiLU,
one-group normalization and pointwise lifting/projection matching PDEquinox
conventions. Inputs are state, scalar transport time, sine and cosine position.
PDEquinox is not imported at runtime.

Both expose integer `dim` and `__call__(self, t, x, *args, rngs=None)` and preserve
flat single/batched shapes. All parameters participate in NNX split/merge,
parameter differentiation and the existing matrix-free NGD algorithm.

## Running

From the repository root:

```bash
JAX_PLATFORMS=cpu python examples/benchmark_runner.py \
  --config examples/configs/kl_ks_smoke.json --run-id 0
# --run-id 1 selects NGD; omit the flag to execute both methods.
JAX_PLATFORMS=cpu python examples/benchmark_runner.py \
  --config examples/configs/kl_acoustic_smoke.json --run-id 1
```

Both commands can use the `neuripp` environment's Python. Full starting
presets are `kl_acoustic.json` and `kl_ks.json`; variants are
`kl_acoustic_128.json`, `kl_acoustic_64.json`, and `kl_ks_64.json`.
They are initial hyperparameters, not calibrated/converged posterior results.
Smoke presets use batch/MC count one, two Euler transport steps, one update,
and at most three CG iterations. The existing `_plots.pdf` retains training
diagnostics; post-run plotting additionally writes `_posterior.pdf` for acoustic
and KS inverse fields. It draws a configurable sample batch from the last saved
weights after training (default 256, in batches of one), computes summaries,
and reuses that batch for the figure. Set `plotting.n_posterior_samples`,
`plotting.posterior_batch_size`, and `plotting.posterior_seed` to tune this
postprocessing cost and reproducibility. The acoustic smoke preset uses 64
draws in batches of four. Run metadata records target settings and installed
dependency versions.

### Notebook postprocessing

Import the helpers from the repository root. The checkpoint helper returns
draws and the neural-ODE log density `log q_theta(x)`; its sampled MAP is the
draw with the highest model log density, **not** the highest true-target
posterior density. For acoustic runs, `samples` and `log_density` are in
log-contrast `x`, while `plot_samples` and `truth` are in physical sound speed
(m/s). Acoustic pixel statistics are computed after transforming every draw.
The ground truth is returned only for visual comparison. The configs use the
Hutchinson divergence estimator, so log densities and the MAP ranking use that
estimator's approximation.

```python
from examples.inverse_pde_visualization import (
    sample_run_posterior,
    summarize_field_samples,
    plot_acoustic_summary,
    plot_ks_summary,
)

draws = sample_run_posterior(
    session_dir, run_id="0", n_samples=256, seed=0, checkpoint="last"
)
summary = summarize_field_samples(draws["plot_samples"], draws["log_density"])
if draws["distribution"]["name"] == "acoustic":
    fig, axes = plot_acoustic_summary(draws["truth"], summary)
else:
    fig, axis = plot_ks_summary(draws["truth"], summary)
```

Use `checkpoint="best"` to load the best checkpoint instead. The summary
function accepts already available `samples` and model `log_density`, so you
can cache and reuse the same draws without sampling again. `plot_run_posterior`
is a convenience function that loads, samples, summarizes, and plots in one call.

## Verification and resource measurements

Focused commands (do not collect all repository training scripts):

```bash
python -m pytest tests/test_pde_inverse_targets.py tests/test_pde_rhs_networks.py \
  tests/test_pde_benchmark_runner.py -q
/usr/bin/time -v python tests/smoke_pde_optimizer.py acoustic adam
/usr/bin/time -v python tests/smoke_pde_optimizer.py acoustic ngd
/usr/bin/time -v python tests/smoke_pde_optimizer.py ks adam --size 200
/usr/bin/time -v python tests/smoke_pde_optimizer.py ks ngd --size 200
```

Measured on the i7-1355U laptop, CPU float32, fresh processes (compilation
included). Full runner tests include plotting and serialization:

| Runner smoke | Peak RSS | Wall time |
|---|---:|---:|
| KS 64, Adam | 1.04 GiB | 21 s |
| KS 64, NGD | 1.38 GiB | 42 s |
| Acoustic 64 square, 4 sources/8 receivers, Adam | 1.25 GiB | 29 s |
| Acoustic 64 square, 4 sources/8 receivers, NGD | 1.53 GiB | 45 s |

Standalone parameter-update checks also passed: acoustic Adam/NGD parameter
delta L2 norms were 0.0714 / 3.44e-7; KS at 200 points with eight sensors gave
0.0333 / 4.12e-6. All checked parameters and step metrics were finite. Tiny
clipped NGD updates are evidence of algorithm compatibility, not convergence.

Tests cover priors, deterministic data, seed independence, reference KS rollout,
gradient finite differences, checkpoint/chunk equivalence with a remainder,
same-shape trace reuse, NNX parameter counts, JVPs and config planning. Float32
acoustic gradient equivalence uses relative vector error; strict componentwise
comparisons were also checked in float64.

The 200-point/20-sensor KS default target is tested. Full 128/256-square acoustic
training and A100 execution remain unverified; memory figures above must not be
extrapolated as guaranteed GPU requirements. Generated test runs live outside
the repository under `/tmp/opencode`, and are not intended for version control.
