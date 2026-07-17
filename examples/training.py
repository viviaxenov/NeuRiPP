import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


import sys

import jax

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
n_restarts = 6
batch_size = 512
N_mc = batch_size
method = sys.argv[1] if len(sys.argv) > 1 else "ngd"
problem = sys.argv[2] if len(sys.argv) > 2 else "checkerboard"
n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else n_iter

rngs = nnx.Rngs(42)

rhs_net = MLP(dim, rngs, dim_hidden=64, n_hidden=1, activation=nnx.swish)

model_args = (
    rhs_net,
    rngs,
    N_mc,
)

model_kwargs = dict(
    ode_nstep_max=6,
    divergence_method="hutchinson",
    ode_method="rk45",
    ode_kwargs=dict(h_max=0.3,  adaptive=True),
)

model = ParametricPushforward(*model_args, **model_kwargs)


match method:
    case "ngd":
        get_fn = get_ngd
        stepsize =  0.001
        args = (stepsize,)
        kwargs = dict(
            linear_solver_regularization=1e-2,
        )
    case "anderson":
        get_fn = partial(get_anderson, history_length=6, natural_grad_clipping_threshold=0.1)
        stepsize = 0.1
        relaxation = 1.0
        reg_factor =  1e-7
        args = (stepsize, relaxation, reg_factor)
        kwargs = dict(
            linear_solver_regularization=1e-1,
            linear_solver_maxiter=100,
        )

    case x if x in optax_optimizers:
        get_fn = partial(get_optax, method=method)
        lr = 1e-4
        args = (lr,)
        kwargs = {"weight_decay": 0.001}
    case _:
        raise ValueError(f"Method {x} not supported")

match problem:
    case "checkerboard":
        data_gen = CheckerboardBatcher( batch_size, 30)
        loss = cross_entropy
    case "two_spirals":
        data_gen = TwoSpiralsBatcher(batch_size, 30)
        loss = cross_entropy
    case "eight_gaussians":
        data_gen = EightGaussiansBatcher(batch_size, 30)
        loss = cross_entropy
    case "st":
        data_gen = LatentBatcherFromModel( batch_size, 1, model)
        logpdf = logpdf_st
        loss = getKL(logpdf)
    case "db":
        data_gen = LatentBatcherFromModel( batch_size, 1, model)
        shift=jnp.array([2.0, 0])
        logpdf = partial(logpdf_double_banana, shift=shift)
        loss = getKL(logpdf)
    case _:
        raise ValueError(f"{problem = } not yet implemented")

init_fun, step_fun = get_fn(loss)
step_fun = nnx.jit(step_fun)
fs = []
grads = []
natural_grads = []

metrics = (fs, grads, natural_grads)


print(f"Running {method.upper()}", flush=True)


batch = data_gen(rngs)
state = init_fun(
    model,
    args,
    kwargs,
    batch, 
    rngs,
)

for _ in tqdm.tqdm(range(n_iter)):
    batch = data_gen(rngs)
    state, vals = step_fun(state, batch, rngs)
    [arr.append(val) for arr, val in zip(metrics, vals)]

model = state[0]

jnp.savez(
    f"single_train_results_{method}.npz",
    grads=grads,
    natural_grads=natural_grads,
    fs=fs,
)

fs = jnp.array(fs)
fs = jnp.where(jnp.isfinite(fs), fs, jnp.inf)
grads = jnp.array(grads)
natural_grads = jnp.array(natural_grads)

x = model.sample(500, rngs)

fig, axs = plt.subplots(1, 3, figsize=(25, 8), layout="constrained")
ax = axs[0]
ax.scatter(
    *x[:, :2].T, label=r"$T_{\text{opt}}^{{{i}}}(z)$", marker="*", s=8.0, zorder=2
)
if problem in ("two_spirals", "eight_gaussians", "checkerboard"):
    ax.scatter(*batch[:, :2].T, label="Data", s=3.0, zorder=1)
elif problem in ("st", "db"):
    x = jnp.linspace(-4, 4, 300, endpoint=True)
    y = jnp.linspace(-4, 4, 300, endpoint=True)
    XX, YY = jnp.meshgrid(x, y, indexing='ij')
    coord = jnp.stack((XX, YY), axis=-1).reshape((-1, 2))
    if problem == "db":
        coord += shift[None, :]
    ax.contour(XX, YY, logpdf(coord).reshape(XX.shape), levels=10, linewidths=0.7, cmap='berlin', alpha=0.5, zorder=1)



ax = axs[1]
ax.plot(fs, label="Loss (best)")

ax = axs[2]
ax.plot(grads, color="red", label=r"$\| \nabla L\|_2$")
if len(natural_grads) > 0:
    ax1 = ax.twinx()
    ax1.plot(natural_grads, color="tab:orange", label=r"$\| \partial_W L\|_2$")
    ax1.set_yscale("log")
ax.set_yscale("log")
fig.suptitle(method.upper())
fig.legend()
fig.savefig(f"test_pm_single_{problem}_{method}.pdf")


# print("Running scan", flush=True)
# t = perf_counter()
# carry = init_fun(model)
# _, metrics = jax.lax.scan(step_fun, carry, length=n_iter_scan)
# dt = perf_counter() - t
# print(f"{method.upper()} scan [1-st run]: {dt=:.2e}, per iter {dt/n_iter_scan:.2e}")
# t = perf_counter()
# carry = init_fun(model)
# _, metrics = jax.lax.scan(step_fun, carry, length=n_iter_scan)
# dt = perf_counter() - t
# print(f"{method.upper()} scan [2-nd run]: {dt=:.2e}, per iter {dt/n_iter_scan:.2e}")
