import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


import sys

import jax

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

def plot_intermediate(model, metrics, iteration: int, method, n_samples=512):
    x = model.sample(n_samples)
    loss, grads, natural_gras = metrics

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 8), layout='constrained')

    ax = axs[0]
    ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}(z)$", marker="*", s=5.0)

    ax = axs[1]
    ax.plot(loss, label="Loss")

    ax = axs[2]
    ax.plot(grads, color="red", label=r"$\| \nabla L\|_2$")
    ax.set_yscale("log")
    if len(natural_grads) > 0:
        ax1 = ax.twinx()
        ax1.plot(natural_grads, color="tab:orange", label=r"$\| \partial_W L\|_2$")
        ax1.set_yscale("log")

    for ax in axs:
        ax.grid()
    fig.suptitle(f"{method.upper()}, iteration {iteration:d}")
    fig.legend()
    return fig

def write_checkpoint(model, output_path):
    pass

def write_intermediate(model, metrics, iteration, method, output_path, n_samples=512):
    fig = plot_intermediate(model, metrics,iteration, method, n_samples=n_samples)
    fig.savefig(os.path.join(output_path, f"test_{method}.pdf"))
    write_checkpoint(model, output_path)




dim = 2
n_iter = 6_000
batch_size = 1024
n_samples_plot = 512
plot_every = 10
N_mc = batch_size
lr = 1e-5
lr_ngd = 0.0001
Lam_reg = 1e-3
method = sys.argv[1] if len(sys.argv) > 1 else "ngd"
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else n_iter

rhs_net = MLP(dim, dim_hidden=32, n_hidden=2, activation=nnx.swish)

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
            loss, lr_ngd, 8, 1.2, 1e-2, "l2", True, Lam_reg, 1e-6, 100
        )
    case x if x in optax_optimizers:
        init_fun, step_fun = get_optax(loss, method, lr)
    case _:
        raise ValueError(f"Method {x} not supported")


os.makedirs(f"./outputs/{method}/last/", exist_ok=True)
os.makedirs(f"./outputs/{method}/best/", exist_ok=True)

step_fun = nnx.jit(step_fun)

best_loss = jnp.inf
fs = []
grads = []
natural_grads = []

metrics = (fs, grads, natural_grads)
carry = init_fun(model, next(data_gen))
for i in tqdm.tqdm(range(n_iter)):
    carry, vals = step_fun(carry, next(data_gen))
    [arr.append(val) for arr, val in zip(metrics, vals)]
    if vals[0] < best_loss:
        best_loss = vals[0]
        write_intermediate(model, metrics, i + 1, method, f"./outputs/{method}/best/", n_samples=n_samples_plot)

    if (i + 1) % plot_every == 0:
        write_intermediate(model, metrics, i + 1, method, f"./outputs/{method}/last/", n_samples=n_samples_plot)

write_intermediate(model, metrics, i, method, f"./outputs/{method}/last/", n_samples=n_samples_plot)


model = carry[0]

