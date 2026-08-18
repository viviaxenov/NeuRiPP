import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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


dim = 2
n_iter = 6_000
n_restarts = 5
n_iter_scan = 100
# batch_size = 2048
batch_size = 512
N_mc = batch_size
lr = 1e-5
lr_ngd = 0.0001
Lam_reg = 1e-3
method = sys.argv[1] if len(sys.argv) > 1 else "ngd"
problem = sys.argv[2] if len(sys.argv) > 2 else "checkerboard"
n_iter = int(sys.argv[3]) if len(sys.argv) > 3 else n_iter

rngs = nnx.Rngs(42)

rhs_net = MLP(dim, rngs, dim_hidden=128, n_hidden=2, activation=nnx.swish)

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


match problem:
    case "checkerboard":
        data_gen = CheckerboardBatcher(batch_size, 30)
        loss = cross_entropy
    case "st":
        data_gen = LatentBatcherFromModel(batch_size, 30, model)
        loss = getKL(logpdf_st)
    case _:
        raise ValueError(f"{problem = } not yet implemented")

match method:
    case "ngd":
        init_fun, step_fun = get_ngd(
            loss,
        )
        stepsizes = jnp.array([0.01, 0.001, 0.0001])
        linear_regs = jnp.array([1e-1, 1e-2, 1e-3, 1e-4])
        # ls_tol = jnp.array([1e-6])
        # ls_maxiter = jnp.array([100], dtype=jnp.int32)
        run_no = jnp.array(range(n_restarts))
        _, SSs, LRs = jnp.stack(
            jnp.meshgrid(run_no, stepsizes, linear_regs), axis=0
        ).reshape(3, -1)
        # _, SSs, LRs, LTols, LMaxs = jnp.stack(
        #     jnp.meshgrid(run_no, stepsizes, linear_regs, ls_tol, ls_maxiter), axis=0
        # ).reshape(5, -1)
        vectorized_args = (SSs,)
        vectorized_kwargs = dict(
            linear_solver_regularization=LRs,
            linear_solver_tolerance=jnp.full_like(SSs, 1e-6),
            linear_solver_maxiter=jnp.full_like(SSs, 100),
        )
        print(vectorized_args)
        print(vectorized_kwargs)
        n_seeds = vectorized_args[0].shape[0]
    case "anderson":
        init_fun, step_fun = get_anderson(
            loss, lr_ngd, 8, 1.2, 1e-2, "l2", True, Lam_reg, 1e-6, 100
        )
    case x if x in optax_optimizers:
        init_fun, step_fun = get_optax(loss, method)
        lr_vals = jnp.array([1e-2, 1e-3, 1e-4, 1e-5])
        lr_repeated = jnp.broadcast_to(
            lr_vals[None, :], (n_restarts, *lr_vals.shape)
        ).ravel()
        vectorized_args = (lr_repeated,)
        n_seeds = lr_repeated.shape[0]
    case _:
        raise ValueError(f"Method {x} not supported")
# step_fun = nnx.jit(step_fun)

fs = []
grads = []
natural_grads = []

metrics = (fs, grads, natural_grads)


print(f"Running {method.upper()}", flush=True)
# vectrozed_args = jnp.stack(jnp.meshgrid(run_no, stepsizes, linear_regs, ls_tol, ls_maxiter), axis=0).reshape(5, -1)[1:]


vectorized_init = nnx.vmap(init_fun)
vectorized_step = nnx.jit(nnx.vmap(step_fun))
vectorized_rngs = rngs.fork(split=n_seeds)

ensemble = nnx.vmap(
    lambda _x: ParametricPushforward(
        MLP(dim, _x, dim_hidden=128, n_hidden=2, activation=nnx.swish),
        _x,
        N_mc,
        **model_kwargs,
    )
)(vectorized_rngs)


state = vectorized_init(
    ensemble,
    *vectorized_args,
    **vectorized_kwargs,
)

for _ in tqdm.tqdm(range(n_iter)):
    batch = data_gen(rngs)
    # train on the same batch for now, later will vectorize batch gen
    batch = jnp.broadcast_to(batch[None, :, :], (n_seeds, *batch.shape))
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

fig, axs = plt.subplots(1, 3)
ax = axs[0]
ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}^{{{i}}}(z)$", marker="*", s=5.0)

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
fig.savefig(f"test_pm_{method}.pdf")

exit()

print("Running scan", flush=True)
t = perf_counter()
carry = init_fun(model)
_, metrics = jax.lax.scan(step_fun, carry, length=n_iter_scan)
dt = perf_counter() - t
print(f"{method.upper()} scan [1-st run]: {dt=:.2e}, per iter {dt/n_iter_scan:.2e}")
t = perf_counter()
carry = init_fun(model)
_, metrics = jax.lax.scan(step_fun, carry, length=n_iter_scan)
dt = perf_counter() - t
print(f"{method.upper()} scan [2-nd run]: {dt=:.2e}, per iter {dt/n_iter_scan:.2e}")
