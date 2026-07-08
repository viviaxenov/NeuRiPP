import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


import sys

import jax

# jax.config.update("jax_debug_infs", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp
import jax.scipy as jsp


from flax import nnx
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.functionals.KL import getKL
from neuripp.functionals.MMD import getMMD
from neuripp.functionals.CrossEntropy import cross_entropy
from neuripp.methods.ngd import get_ngd
from neuripp.methods.anderson import get_anderson
from neuripp.methods.optax_optimizer import get_optax, optax_optimizers
from neuripp.utility.utility import *

from time import perf_counter

import matplotlib.pyplot as plt

from rhs_architectures import LinearRHS, MLP
from data_generators import *
from logpdf_targets import *

import tqdm
from functools import partial


dim = 2
n_iter = 6_000
n_restarts = 5
batch_size = 512
N_mc = batch_size

method = sys.argv[1] if len(sys.argv) > 1 else "ngd"
problem = sys.argv[2] if len(sys.argv) > 2 else "checkerboard"
n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else n_iter

rngs = nnx.Rngs(42)

# rhs_net = MLP(dim, rngs, dim_hidden=128, n_hidden=2, activation=nnx.swish)
rhs_net = MLP(dim, rngs, dim_hidden=16, n_hidden=1, activation=nnx.swish)
model_args = (
    rhs_net,
    rngs,
    N_mc,
)
model_kwargs = dict(
    ode_nstep_max=12,
    divergence_method="hutchinson",
    # ode_method="euler",
    ode_method="rk45",
    ode_kwargs=dict(h_max=0.3, N_iter_to_accept=15, adaptive=True),
)
model = ParametricPushforward(*model_args, **model_kwargs)


match method:
    case "ngd":
        n_restarts = 40
        get_fn = partial(get_ngd, natural_grad_clipping_threshold=10.)
        stepsizes = jnp.array([0.01, 0.001, 0.0001])
        linear_regs = jnp.array([ 1e-3 ])
        run_no = jnp.array(range(n_restarts))
        _, SSs, LRs = jnp.stack(
            jnp.meshgrid(run_no, stepsizes, linear_regs), axis=0
        ).reshape(3, -1)
        vectorized_args = (SSs,)
        vectorized_kwargs = dict(
            linear_solver_regularization=LRs,
        )
    case "anderson":
        # get_fn = partial(get_anderson, history_length=6, natural_grad_clipping_threshold=10.)
        get_fn = partial(get_anderson, history_length=6, )
        stepsizes = jnp.array([0.01])
        relaxations = jnp.array([1.0])
        reg_factors = jnp.array([1e-1, 1e-3, 1e-5])
        linear_regs = jnp.array([1e-2])
        run_no = jnp.array(range(n_restarts))
        _, SSs, Betas, Gammas, LRs = jnp.stack(
            jnp.meshgrid(run_no, stepsizes, relaxations, reg_factors, linear_regs),
            axis=0,
        ).reshape(5, -1)
        vectorized_args = (SSs, Betas, Gammas)
        vectorized_kwargs = dict(
            linear_solver_regularization=LRs,
            linear_solver_maxiter=jnp.full_like(LRs, 50),
        )

    case x if x in optax_optimizers:
        n_restarts = 30
        get_fn = partial(get_optax, method=method)
        lr_vals = jnp.array([1e-2, 1e-3, 1e-4, 1e-5])
        lr_repeated = jnp.broadcast_to(
            lr_vals[None, :], (n_restarts, *lr_vals.shape)
        ).ravel()
        vectorized_args = (lr_repeated,)
        wd = jnp.full_like(lr_repeated, 0.002)
        wd = wd.at[::2].set(0.00001)
        vectorized_kwargs = {"weight_decay": wd}
    case _:
        raise ValueError(f"Method {x} not supported")

n_lanes = vectorized_args[0].shape[0]
print(f"Training in parallel, ensemble size {n_lanes}")
match problem:
    case "checkerboard":
        data_gen = CheckerboardBatcher((n_lanes, batch_size), 30)
        loss = cross_entropy
    case "two_spirals":
        data_gen = TwoSpiralsBatcher((n_lanes, batch_size), 30)
        loss = cross_entropy
    case "eight_gaussians":
        data_gen = EightGaussiansBatcher((n_lanes, batch_size), 30)
        loss = cross_entropy
    case "st":
        data_gen = LatentBatcherFromModel((n_lanes, batch_size), 1, model)
        logpdf = logpdf_st
        loss = getKL(logpdf)
    case "db":
        data_gen = LatentBatcherFromModel((n_lanes, batch_size), 1, model)
        shift = jnp.array([2.0, 0])
        logpdf = partial(logpdf_double_banana, shift=shift)
        loss = getKL(logpdf)
    case _:
        raise ValueError(f"{problem = } not yet implemented")

init_fun, step_fun = get_fn(loss)
fs = []
grads = []
natural_grads = []

metrics = (fs, grads, natural_grads)


print(f"Running {method.upper()}", flush=True)

vectorized_init = nnx.vmap(init_fun)
vectorized_step = nnx.jit(nnx.vmap(step_fun))
vectorized_rngs = rngs.fork(split=n_lanes)

# Initialize multiple models
ensemble = nnx.vmap(
    lambda _x: ParametricPushforward(
        MLP(dim, _x, dim_hidden=16, n_hidden=1, activation=nnx.swish),
        _x,
        N_mc,
        **model_kwargs,
    )
)(vectorized_rngs)

# Ensure every model in the ensemble starts with the same params
gd, ens_params, rest = nnx.split(ensemble, nnx.Param, ...)
ens_params = jax.tree.map(
    lambda _leaf: jnp.broadcast_to(_leaf[:1, ...], _leaf.shape), ens_params
)
ensemble = nnx.merge(gd, ens_params, rest)

batch = data_gen(rngs)
state = vectorized_init(
    ensemble,
    vectorized_args,
    vectorized_kwargs,
    batch,
    vectorized_rngs,
)

for _ in tqdm.tqdm(range(n_iter)):
    batch = data_gen(rngs)
    state, vals = vectorized_step(state, batch, vectorized_rngs)
    [arr.append(val) for arr, val in zip(metrics, vals)]

model_ensemble = state[0]

match method:
    case "ngd":
        jnp.savez(
            f"ensemble_train_results_{method}.npz",
            step_sizes=SSs,
            linear_solver_regs=LRs,
            grads=grads,
            natural_grads=natural_grads,
            fs=fs,
        )
    case x if x in optax_optimizers:
        jnp.savez(
            f"ensemble_train_results_{method}.npz",
            learning_rates=lr_repeated,
            grads=grads,
            fs=fs,
        )


fs = jnp.array(fs)
fs = jnp.where(jnp.isfinite(fs), fs, jnp.inf)
grads = jnp.array(grads)
natural_grads = jnp.array(natural_grads)

gd, _, rest = nnx.split(model, nnx.Param, ...)
_, param_ensemble, _ = nnx.split(model_ensemble, nnx.Param, ...)
i = jnp.argmin(fs[-1, :])
print(fs[-1, i])
param = jax.tree.map(lambda x: x[i, ...], param_ensemble)
x = nnx.merge(gd, param, rest).sample(500, rngs)
loss_mean = fs[:, i // n_restarts : i // n_restarts + n_restarts].mean(axis=-1)
loss_std = fs[:, i // n_restarts : i // n_restarts + n_restarts].std(axis=-1)

fig, axs = plt.subplots(1, 3, figsize=(25, 8), layout="constrained")
ax = axs[0]
ax.scatter(
    *x[:, :2].T, label=r"$T_{\text{opt}}^{{{i}}}(z)$", marker="*", s=8.0, zorder=2
)
if problem in ("two_spirals", "eight_gaussians", "checkerboard"):
    one_batch = batch[0, :, :]
    ax.scatter(*one_batch[:, :2].T, label="Data", s=3.0, zorder=1)
elif problem in ("st", "db"):
    x = jnp.linspace(-4, 4, 300, endpoint=True)
    y = jnp.linspace(-4, 4, 300, endpoint=True)
    XX, YY = jnp.meshgrid(x, y, indexing="ij")
    coord = jnp.stack((XX, YY), axis=-1).reshape((-1, 2))
    if problem == "db":
        coord += shift[None, :]
    ax.contour(
        XX,
        YY,
        logpdf(coord).reshape(XX.shape),
        levels=10,
        linewidths=0.7,
        cmap="berlin",
        alpha=0.5,
        zorder=1,
    )


ax = axs[1]
ax.plot(range(fs.shape[0]), fs[:, i], label="Loss (best)")
ax.fill_between(
    range(fs.shape[0]),
    loss_mean + 3.0 * loss_std,
    loss_mean - 3.0 * loss_std,
    alpha=0.2,
)

ax = axs[2]
ax.plot(grads[:, i], color="red", label=r"$\| \nabla L\|_2$")
if len(natural_grads) > 0:
    ax1 = ax.twinx()
    ax1.plot(natural_grads[:, i], color="tab:orange", label=r"$\| \partial_W L\|_2$")
    ax1.set_yscale("log")
ax.set_yscale("log")
fig.suptitle(method.upper())
fig.legend()
fig.savefig(f"test_pm_{problem}_{method}.pdf")
