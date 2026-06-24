import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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
from neuripp.functionals.KL import getKL, logpdf_st
from neuripp.functionals.MMD import *
from neuripp.functionals.CrossEntropy import cross_entropy
from neuripp.methods.ngd import get_ngd
from neuripp.methods.sgd import get_sgd
from neuripp.methods.anderson import get_anderson
from neuripp.methods.optax_optimizer import get_optax, optax_optimizers
from neuripp.utility.utility import *

from time import perf_counter

import matplotlib.pyplot as plt

from test_rhs import LinearRHS, MLP

import tqdm


dim = 2
n_iter = 5_000
n_iter_scan = 100
batch_size = 2048
N_mc = batch_size
lr = 1e-5
lr_ngd = 0.0001
Lam_reg = 1e-3
method = sys.argv[1] if len(sys.argv) > 1 else "ngd"
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else n_iter

rhs_net = MLP(dim, dim_hidden=128, n_hidden=2, activation=nnx.swish)

model_args = (
    rhs_net,
    N_mc,
    42,
)
model_kwargs = dict(
    ode_nstep_max=12,
    divergence_method="hutchinson",
    # ode_method="euler",
    ode_method="rk45",
    ode_kwargs=dict(h_max=0.3, N_iter_to_accept=15, adaptive=True),
)

data_gen = checkerboard_generator(batch_size, 30)
loss = cross_entropy
# loss = getMMD(batch_size, next(data_gen), jnp.array([0.025, 0.25, 2.5]))
# loss = getKL(logpdf_st, batch_size)

vg_fn = nnx.value_and_grad(loss, has_aux=True)

print(f"Running {method.upper()}", flush=True)
rhs_net = MLP(dim, dim_hidden=128, n_hidden=2, activation=nnx.swish)
gd, par_init, rest = nnx.split(rhs_net, nnx.Param, ...)
par_init = jax.tree.map(lambda _x: jnp.full_like(_x, 1e-9), par_init)
nnx.update(rhs_net, par_init)
model = ParametricPushforward(*model_args, **model_kwargs)

match method:
    case "ngd":
        init_fun, step_fun = get_ngd(
            loss,
            step_size=lr_ngd,
            linear_solver_regularization=Lam_reg,
            linear_solver_tolerance=1e-6,
            linear_solver_maxiter=100,
        )
    case "sgd":
        init_fun, step_fun = get_sgd(loss, step_size=lr)
    case "anderson":
        init_fun, step_fun = get_anderson(
            loss, lr_ngd, 8, 1.0, 1e-3, "l2", True, Lam_reg, 1e-6, 100
        )
    case x if x in optax_optimizers:
        init_fun, step_fun = get_optax(loss, method, lr)
    case _:
        raise ValueError(f"Method {x} not supported")


step_fun = nnx.jit(step_fun)

fs = []
grads = []
natural_grads = []

metrics = (fs, grads, natural_grads)

carry = init_fun(model, next(data_gen))
for _ in tqdm.tqdm(range(n_iter)):
    carry, vals = step_fun(carry, next(data_gen))
    [arr.append(val) for arr, val in zip(metrics, vals)]

model = carry[0]

x = model.sample(3000)

fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}(z)$", marker="*", s=5.0)

ax = axs[1]
ax.plot(fs, label="Loss")
ax1 = ax.twinx()
ax1.plot(grads, color="red", label=r"$\| \nabla L\|_2$")
if len(natural_grads) > 0:
    ax1.plot(natural_grads, color="tab:orange", label=r"$\| \partial_W L\|_2$")
ax.set_yscale("log")
ax1.set_yscale("log")
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
