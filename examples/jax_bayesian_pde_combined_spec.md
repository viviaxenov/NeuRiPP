# JAX Bayesian PDE inverse problems: reproduction specifications

This document combines the complete reproduction specifications for the j-Wave acoustic full-waveform inversion benchmark and the Exponax Kuramoto–Sivashinsky initial-condition inversion benchmark. References are numbered independently within each section.

## j-Wave: Bayesian acoustic full-waveform inversion


**Status:** Defined benchmark adapted from published components; **not** a previously published end-to-end Bayesian j-Wave experiment. All choices below are fixed for reproducibility.

### 1. Unknown, observations, and inverse problem

- Unknown: a real-valued **2D log-velocity-contrast field** `x` of shape `(256, 256)` (the parameter vector, flattened when computing densities). The physical sound-speed field supplied to j-Wave is
  \[
  c(x)_{ij}=1480\,[1+0.10\,M_{ij}\tanh(x_{ij}/0.10)]\ \mathrm{m/s},
  \]
  where `M = circ_mask((256,256), 80, (128,128))`. Thus, `c=1480 m/s` outside the reconstruction circle and `1332<c<1628 m/s` inside. The bounded, differentiable transform follows the *principle* of the j-Wave FWI tutorial's constrained sound-speed parameterization; the exact transform here is a benchmark choice. **Refs:** [1, 2].
- Forward map `F(x)`: launch a known acoustic pulse from each selected source, numerically propagate pressure through medium `c(x)`, and record **raw pressure time-series** at receivers. Observations `y` have shape `(64, Nt, 64)` in **(source, time, receiver)** order; `Nt` is determined once by the fixed time axis. **Refs:** [1, 2].
- Posterior target (constants omitted):
  \[
  \log p(x\mid y)=-\tfrac12\|L^{-1}x\|_2^2-\tfrac1{2\sigma_n^2}\|F(x)-y\|_2^2,
  \quad x\in\mathbb R^{256\times256}.
  \]
  `L` is the prior square-root covariance operator specified below. **Refs:** [3, 4].

### 2. Prior and synthetic ground truth

Use a **mean-zero stationary correlated Gaussian field** `x ~ N(0, LLᵀ)` as the inference prior, modeling smooth spatial variation rather than independent pixel velocities. Construct it using a 2D FFT on the periodic grid:

\[
\xi_{ij}\overset{\mathrm{iid}}{\sim}N(0,1),\quad
h(k)=\left[1+\ell^2\lvert 2\pi k\rvert^2\right]^{-1},\quad
L\xi=\sigma_x\,\mathrm{Re}\left[\mathrm{IFFT2}\left(\frac{h}{\sqrt{\mathrm{mean}(h^2)}}\,\mathrm{FFT2}(\xi)\right)\right].
\]

Fixed prior hyperparameters: `ell = 1.0e-3 m`, `sigma_x = 0.04` (dimensionless log-contrast units); `k` is the 2D frequency grid from `jnp.fft.fftfreq(256, d=0.1e-3)` (cycles/m). This specifies a Matérn-like smooth Gaussian-field prior. Evaluate `L⁻¹x` by dividing its Fourier coefficients by the **same positive filter**; use the resulting norm in the log prior. **The synthetic truth is not a draw from this prior.** It is the deterministic sound-speed phantom from the j-Wave FWI tutorial:

```python
C1 = circ_mask((256, 256), 16, (100, 100))
C2 = circ_mask((256, 256), 20, (165, 128))
C3 = circ_mask((256, 256), 40, (130, 130))
c_true = 1480 * (1 + 0.05*C1 + 0.02*C2 + 0.03*C3)
```

For consistency with the inferred parameterization, convert this physical truth to `x_true` by inverting the bounded map inside `M`:

```python
safe_M = jnp.where(M > 0, M, 1.0)
r = jnp.where(M > 0, (c_true / 1480 - 1) / (0.10 * safe_M), 0.0)
x_true = 0.10 * jnp.arctanh(r)
```

In implementation, use a safe denominator outside `M`; the three circles all lie
inside the reconstruction mask and their maximum contrast is strictly below
the transform bound. The truth is fixed and independent of `truth_seed`; the
inference prior remains the Gaussian field above. The three-circle geometry,
amplitudes and reference speed are copied from the j-Wave FWI tutorial [2]; the
log-contrast inverse mapping is specific to this benchmark. **Refs:** [2–4].

### 3. JAX forward solver: j-Wave

Installation (Python 3.11+; choose either CPU or the appropriate supported JAX GPU backend):

```bash
python -m pip install -U jax              # CPU; follow JAX docs for GPU builds instead
python -m pip install jwave
python -m pip freeze > requirements-lock.txt
```

Use `from jwave.acoustics.time_varying import simulate_wave_propagation`, `jwave.FourierSeries`, and `jwave.geometry.{Domain, Medium, Sources, Sensors, TimeAxis, points_on_circle, circ_mask}`. **Refs:** [1, 2, 5].

| Fixed setting | Value |
|---|---|
| Grid / spacing | `Domain((256,256), (0.1e-3,0.1e-3))` |
| Medium | `Medium(..., sound_speed=FourierSeries(c[...,None], domain), pml_size=20)`; other medium parameters at defaults |
| Time axis | **Construct once**, not from trial `x`: `TimeAxis.from_medium(Medium(domain, sound_speed=1630.0), cfl=0.25, t_end=25e-6)` |
| Sources / receivers | 64 colocated positions from `points_on_circle(64, 100, (128,128))` |
| Source pulse | Copy the j-Wave FWI tutorial exactly: `source_freq=1e6`, `source_mag=1.3e-5 / time_axis.dt`, `apply_ramp`, then `gaussian_window(..., center=3e-6, width=1.2e-6)` |
| PDE execution | For each source, `simulate_wave_propagation(medium, time_axis, sources=sources, sensors=sensors, checkpoint=True)`; batch source indices with `jax.vmap` (or loop to reduce peak memory) |
| Differentiation | Keep `time_axis`, sources, receivers, and geometry fixed across evaluations; require `jax.jit(F)` and `jax.grad(log_posterior)` to work |

The 1630 m/s reference speed is strictly above the bounded maximum 1628 m/s; a fixed CFL timestep therefore does not depend on `x`. Acquisition geometry and pulse conventions follow the published j-Wave FWI tutorial; `t_end=25 μs` and the fixed reference speed are explicit benchmark choices. **Refs:** [1, 2, 5].

### 4. Observation noise and likelihood

Generate exactly one dataset:

\[
y_{s,t,r}=F(x_{\mathrm{true}})_{s,t,r}+\epsilon_{s,t,r},\quad
\epsilon_{s,t,r}\overset{\mathrm{iid}}{\sim}N(0,\sigma_n^2),\quad
\sigma_n=0.02\,\operatorname{median}_{s,r}\big(\max_t|F(x_{\mathrm{true}})_{s,t,r}|\big).
\]

Draw noise once with `jax.random.PRNGKey(43)`. Keep `σ_n` fixed and known during inference. Use the **raw traces**, not Hilbert envelopes: the resulting log likelihood is exactly `-0.5 * sum(((F(x)-y)/sigma_n)**2)` (up to constants). The 2%-of-median-trace-peak noise scale and diagonal Gaussian likelihood are adopted from published Bayesian FWI experiments. **Ref:** [6].

### 5. Neural ODE vector field: ~0.51M-parameter 2D U-Net

The existing generator is unchanged:

\[
\dot X_t=v_\theta(X_t,t),\qquad X_0=z,\quad z\sim N(0,I_{256^2}),\qquad T_\theta(z)=X_1.
\]

Use a **time-conditioned residual U-Net** with `v_theta: (B,1,256,256), t -> (B,1,256,256)`:

| Component | Exact proposed configuration |
|---|---|
| Input/output | 1 channel → 1 channel; 3×3 convolutions, padding 1; linear output |
| Encoder | Initial 3×3 `1→16`; one ResBlock at each width `16,32,64`; 3×3 stride-2 downsampling `16→32`, `32→64` |
| Bottleneck | Two ResBlocks `64→64` |
| Decoder | Nearest-neighbor upsampling; concatenate corresponding encoder skips; ResBlocks `128→64`, `96→32`, `48→16` |
| ResBlock | GroupNorm (8 groups) → SiLU → 3×3 conv → **add projected time embedding** → GroupNorm → SiLU → 3×3 conv; 1×1 shortcut when input/output widths differ |
| Time conditioning | 64-dimensional sinusoidal encoding of continuous `t∈[0,1]` → Dense(128) → SiLU → Dense(128); separate Dense(128→channels) projection for every ResBlock |
| Attention | None |
| Final layer | 3×3 conv `16→1`, no final activation |
| Parameter budget | **Approximately 509K** parameters (with convolution/dense biases and GroupNorm affine parameters; exact library count may vary) |

This is a **downsized architecture proposal**, not a claim that a 500K U-Net was used in the cited papers. Spatial U-Nets with time embeddings/residual blocks are established in diffusion models; a considerably larger time-conditioned U-Net is used for a related FWI transport/update network. **Refs:** [7, 8, 9].

#### Literature-scale acoustic comparison preset

For a high-capacity comparison, Huang & Perdikaris [9, Secs. 2.4 and 3.1]
report a guided-diffusion-style timestep-conditioned U-Net with base width 64,
channel multipliers `(1,1,2,3,4)` for their other velocity models, four residual
blocks per resolution, a 256-dimensional time embedding, GroupNorm, and SiLU.
They report approximately 34.70M parameters. Their description does not fix the
attention placement; the NeuRiPP `AcousticGuidedUNetRHS` adaptation places spatial
attention at the 192-channel level and in the bottleneck and has 34,697,025
parameters. It uses four encoder blocks and five skip-consuming decoder blocks
per level; the extra decoder block and 32-group normalization are implementation
choices. The adaptation is used at input resolutions 64, 128, and 256; its
parameter count is independent of spatial resolution. It is not a claim of
exact reproduction or demonstrated optimality for Bayesian posterior transport.

### References

1. Stanziola et al., *j-Wave: An Open-Source Differentiable Wave Simulator* (2022). https://arxiv.org/abs/2207.01499
2. j-Wave, *Full Wave Inversion* tutorial (code and acquisition geometry). https://ucl-bug.github.io/jwave/notebooks/time_varying/FWI.html
3. Stuart, *Inverse Problems: A Bayesian Perspective* (2010). https://doi.org/10.1017/S0962492910000061
4. Gouveia & Scales, *Bayesian Seismic Waveform Inversion: Parameter Estimation and Uncertainty Analysis* (1998). https://doi.org/10.1029/97JB02933
5. j-Wave installation/README. https://github.com/ucl-bug/jwave
6. Zhang et al., *3-D Bayesian Variational Full Waveform Inversion* (2023), particularly the diagonal-Gaussian likelihood and 2% noise convention. https://academic.oup.com/gji/article/234/1/546/7031752
7. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015). https://arxiv.org/abs/1505.04597
8. Ho et al., *Denoising Diffusion Probabilistic Models* (2020), time-conditioned convolutional U-Net design. https://arxiv.org/abs/2006.11239
9. Huang & Perdikaris, *Self-Flow-Matching assisted Full Waveform Inversion* (2026), FWI velocity-network precedent (not Bayesian posterior sampling). https://arxiv.org/abs/2603.13425

---

## Exponax: Bayesian Kuramoto–Sivashinsky inversion


**Status:** Proposed reproducible benchmark assembled from documented components, **not** a previously published end-to-end Exponax Bayesian benchmark. Numerical hyperparameters marked *benchmark choice* are fixed here for replication.

### 1. Unknown, measurements, and PDE

- **Unknown:** the physical initial condition `x = u(·,0)` on a periodic 1D grid, represented as a real JAX array of shape **`(1, 200)`** (one channel, 200 positions). Do not confuse `x` with the Neural ODE's independent standard-normal latent `z`. **Refs:** [1, 2].
- **Forward equation:** conservative 1D Kuramoto–Sivashinsky,
  \[
  u_t + \tfrac12\partial_r(u^2)+u_{rr}+u_{rrrr}=0,
  \qquad r\in[0,100),\quad u(r+100,t)=u(r,t).
  \]
  The forward map `F(x)` advances the solution from `u(·,0)=x` and extracts state values at fixed space-time sensors. **Refs:** [1, 2, 3].
- **Observations:** 20 grid sensors at spatial indices `0,10,…,190`, at physical times **`5, 10, 15, 20`**; `y=F(x_true)+ε` has shape **`(4, 20)`**. Acquisition times/locations are *benchmark choices*, following sparse noisy state-observation setups in data assimilation and KS inference. **Refs:** [3, 4].

### 2. Prior and ground truth

Use a **centered, spatially correlated Gaussian random-field prior** `x ~ N(0, LLᵀ)`, rather than independent Gaussian pixels. This is the standard Bayesian regularization of an ill-posed initial-state inverse problem; Exponax itself implements spectral Gaussian random-field initial conditions. **Refs:** [1, 3, 5].

Define `L` on the periodic 200-point grid via FFT (all frequencies have positive variance):

\[
\begin{aligned}
\xi_j&\overset{\rm iid}{\sim}N(0,1),\\
\omega_k&=2\pi\,\mathrm{fftfreq}(200,\,d=100/200)_k,\\
h_k&=(1+(\ell\omega_k)^2)^{-\alpha/2},\quad
\widetilde h_k=h_k/\sqrt{200^{-1}\sum_j h_j^2},\\
L\xi&=\sigma_x\,\operatorname{Re}\mathrm{IFFT}(\widetilde h\cdot\mathrm{FFT}(\xi)).
\end{aligned}
\]

**Fixed benchmark choices:** `ell = 2.0`, `alpha = 2.0`, `sigma_x = 1.0`, prior mean zero. Draw once using `jax.random.PRNGKey(42)` and set `x_true = Lξ`. Define `L⁻¹x` using the reciprocal of the **same** FFT multiplier; thus, up to an additive constant,
\[
\log p_{\rm prior}(x)=-\tfrac12\|L^{-1}x\|_2^2.
\]
The smoothness/scale choices are **not** claimed to be numerically calibrated in the cited papers. Avoid `zero_mean=True` or per-realization `std_one=True` normalization if using Exponax's `GaussianRandomField` helper to implement this density: these operations change the full-rank Gaussian law. **Refs:** [3, 5].

### 3. Differentiable forward solver: Exponax

**Installation** (Python 3.10+; CPU JAX shown, consult JAX installation instructions for GPU):

```bash
python -m pip install -U 'jax[cpu]' exponax pdequinox
python -m pip freeze > requirements-lock.txt
```

Use the **documented conservative KS stepper** and fixed observation operator:

```python
import jax
import jax.numpy as jnp
import exponax as ex

stepper = ex.stepper.KuramotoSivashinskyConservative(
    num_spatial_dims=1,
    domain_extent=100.0,
    num_points=200,
    dt=0.1,
    order=2,
)
t_idx = jnp.array([49, 99, 149, 199])  # times 5, 10, 15, 20
r_idx = jnp.arange(0, 200, 10)          # 20 sensors

def F(x):                               # x: (1, 200)
    traj = ex.rollout(stepper, 200, include_init=False)(x)
    return traj[t_idx, 0][:, r_idx]      # (4, 20)

F = jax.jit(F)
```

The solver is JAX-native and uses Fourier spectral discretization with exponential time differencing; the specified `order=2` matches Exponax's documented default. Keep all solver settings, sensors, and times fixed when evaluating `F` or differentiating the posterior. **Refs:** [1, 2, 6].

### 4. Noise and log posterior

Generate **one fixed synthetic dataset** with independent additive Gaussian sensor noise:

\[
y=F(x_{\rm true})+\varepsilon,\qquad
\varepsilon_{ij}\overset{\rm iid}{\sim}N(0,\sigma_n^2),\qquad
\boxed{\sigma_n=0.1}.
\]

Draw the noise once with `jax.random.PRNGKey(43)`; the dimensionless SD `0.1` is a *benchmark choice*, not a published KS noise calibration. The diagonal Gaussian likelihood follows the standard state-observation/data-assimilation model. **Refs:** [3, 4].

With fixed known `sigma_n`, the target for **physical** `x` is, up to constants,

\[
\boxed{\log p(x\mid y)=-\tfrac12\|L^{-1}x\|_2^2
                    -\tfrac1{2(0.1)^2}\|F(x)-y\|_F^2.}
\]

Require `jax.jit(log_posterior)` and `jax.grad(log_posterior)` to work. For reproducibility keep the initial condition, noise, and model parameters fixed once generated.

### 5. Neural ODE RHS: ~111K-parameter 1D dilated ResNet

The user's generative map is **unchanged**:
\[
\dot X_\tau=v_\theta(X_\tau,\tau),\qquad
X_0=z\sim N(0,I_{200}),\qquad T_\theta(z)=X_1.
\]
Here `X_τ` has the solver-compatible shape `(1,200)`, and **`τ` is the generative transport time**, not physical KS time. Use a small circular-convolution ResNet rather than a large U-Net. The architectural family is used for chaotic PDE emulation, including KS; the specific use as a posterior-transport vector field is **an adaptation**. **Refs:** [7, 8, 9].

| Component | Fixed configuration |
|---|---|
| Native network | `pdequinox.arch.DilatedResNet` (JAX/Equinox) |
| Input to network | Four `(1,200)` channels concatenated: `X_τ`, constant `τ`, `sin(2πj/200)`, `cos(2πj/200)` |
| Output | One `(1,200)` channel, interpreted as `v_theta(X_τ, τ)` |
| Hidden width / blocks | `hidden_channels=32`, `num_blocks=5` |
| Each block | Seven 1D convolutions, kernel 3, dilations `(1,2,4,8,4,2,1)`, residual skip |
| Nonlinearity / normalization | `jax.nn.silu`, `use_norm=True` (group normalization) |
| Boundary condition | `boundary_mode="periodic"` |
| Attention / downsampling | None |
| Trainable parameters | **111,073**, by PDEquinox's documented counting formula for 4 input channels, 1 output channel, 32 hidden channels, 5 blocks; verify with `pdequinox.count_parameters(model)` |

For an architecture-size comparison, the repository also exposes a 48-channel,
four-block preset inspired by the published Dilated ResNet configuration of
Stachenfeld et al. [7]. The NeuRiPP adaptation retains time and sine/cosine
coordinate input channels, SiLU, and one-group normalization; its exact parameter
count is 197,857. This is a larger posterior-transport RHS variant, not a claim
that the paper's supervised learned simulator was used for Bayesian inversion.

Instantiate using:

```python
import pdequinox as pdeqx

net = pdeqx.arch.DilatedResNet(
    1, 4, 1,
    hidden_channels=32,
    num_blocks=5,
    dilation_rates=(1, 2, 4, 8, 4, 2, 1),
    use_norm=True,
    activation=jax.nn.silu,
    boundary_mode="periodic",
    key=jax.random.PRNGKey(44),
)
```

**Why the coordinate channels?** A periodic CNN driven only by a stationary Gaussian base and a spatially constant time channel would impose translation invariance on generated samples, which a posterior conditioned on fixed-location measurements need not satisfy. The periodic sine/cosine coordinate channels allow position-dependent transport while preserving the grid's periodic geometry; they are an **implementation choice**, motivated by coordinate-aware CNNs. **Ref:** [10].

### References

1. Exponax, *Getting Started / 1D KS example*: https://fkoehler.site/exponax/
2. Exponax, *Conservative Kuramoto–Sivashinsky API*: https://fkoehler.site/exponax/api/stepper/nonlinear/ks_cons/
3. Law & Stuart, *Evaluating Data Assimilation Algorithms* (2012), Bayesian initial-state estimation with correlated Gaussian prior and noisy observations: https://doi.org/10.1175/MWR-D-11-00257.1
4. Parikh et al., *D-Flow SGLD: Source-Space Posterior Sampling for Scientific Inverse Problems with Flow Matching* (2026), sparse KS measurements and generative transport: https://arxiv.org/abs/2602.21469
5. Exponax, *GaussianRandomField API*: https://fkoehler.site/exponax/api/utilities/initial_conditions/gaussian_random_field/
6. Koehler et al., *APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs* (NeurIPS 2024), differentiable spectral PDE solver: https://arxiv.org/abs/2411.00180
7. Stachenfeld et al., *Learned Coarse Models for Efficient Turbulence Simulation* (ICLR 2022), dilated residual CNNs on KS/chaotic flows: https://arxiv.org/abs/2112.15275
8. PDEquinox, *DilatedResNet API*: https://fkoehler.site/pdequinox/api/arch/dilated/
9. PDEquinox, *Parameter Counting & Receptive Field*: https://fkoehler.site/pdequinox/examples/parameter_count_and_receptive_field/
10. Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution* (2018): https://arxiv.org/abs/1807.03247
