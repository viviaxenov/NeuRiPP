import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


import sys
import traceback
import argparse

import jax

jax.config.update("jax_debug_infs", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_log_compiles", True)
import jax.numpy as jnp
import jax.scipy as jsp


from flax import nnx
import functools
from neuripp._ode._ode import *
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.functionals.KL import getKL, logpdf_st
from neuripp.methods.ngd import get_ngd
from neuripp.methods.sgd import get_sgd
from neuripp.utility.utility import *

from time import perf_counter

import matplotlib.pyplot as plt

from test_rhs import LinearRHS, MLP
from operator import add

import tqdm


dim = 50
n_iter = 6_000
batch_size = 2048
N_mc = 2048
lr = 1e-3
lr_ngd = 0.0003
Lam_reg = 1e-3

rhs_net = MLP(dim, dim_hidden=128, n_hidden=2, activation=nnx.tanh)

model_args = (
    rhs_net,
    N_mc,
    42,
)
model_kwargs = dict(
    ode_nstep_max=10,
    divergence_method="hutchinson",
    ode_method="euler",
)


loss = getKL(logpdf_st, batch_size)

vg_fn = nnx.value_and_grad(loss, has_aux=True)

print("Running natural gradient descent", flush=True)
rhs_net = MLP(dim, dim_hidden=126, n_hidden=2, activation=nnx.tanh)
gd, par_init, rest = nnx.split(rhs_net, nnx.Param, ...)
par_init = jax.tree.map(jnp.zeros_like, par_init)
nnx.update(rhs_net, par_init)
model_ngd = ParametricPushforward(*model_args, **model_kwargs)


init_ngd, ngd_step = get_ngd(loss, lr_ngd, Lam_reg, 1e-4, 100)

ngd_step = nnx.jit(ngd_step)

t = perf_counter()
carry = init_ngd(model_ngd)
_, (fs, grads, natural_grads) = jax.lax.scan(ngd_step, carry, length=10)
dt = perf_counter() - t

fs = []
natural_grads = []
grads = []

metrics = (fs, grads, natural_grads)

for _ in tqdm.tqdm(range(n_iter)):
    (model_ngd,),  vals = ngd_step((model_ngd,))
    [arr.append(val) for arr, val in zip(metrics, vals)]

x = model_ngd.sample(3000)

print(x.mean(axis=0))
print(jnp.cov(x, rowvar=False))

fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}(z)$", marker="*", s=5.0)

ax = axs[1]
ax.plot(fs, label="Loss")
ax1 = ax.twinx()
ax1.plot(grads, color="red", label=r"$\| \nabla L\|_2$")
ax1.plot(natural_grads, color="tab:orange", label=r"$\| \partial_W L\|_2$")
ax1.set_yscale("log")
fig.legend()
fig.savefig("test_pm.pdf")


rhs_net = MLP(dim, dim_hidden=32, n_hidden=2, activation=nnx.tanh)
gd, par_init, rest = nnx.split(rhs_net, nnx.Param, ...)
par_init = jax.tree.map(jnp.zeros_like, par_init)
nnx.update(rhs_net, par_init)
model = ParametricPushforward(*model_args, **model_kwargs)

vg_fn = nnx.jit(nnx.value_and_grad(loss, has_aux=True))
t = perf_counter()
(f, (x, logpdf)), grad = vg_fn(model)
dt = perf_counter() - t
print(f"Loss first compute {dt=:.2e}")


fs = []
grad_norms_sq = []


print("Running Euclidean gradient descent", flush=True)
t = perf_counter()
for k in tqdm.tqdm(range(n_iter)):
    (f, (x, logpdf)), grad = vg_fn(model)

    gd, params, rest = nnx.split(model, nnx.Param, ...)
    params_new = jax.tree.map(lambda x, y: x - y * lr, params, grad)
    nnx.update(model, params_new)
    fs.append(f)

    grad_norm_sq = tree_dot_product(grad, grad)
    grad_norms_sq.append(grad_norm_sq)

fs = jnp.array(fs)
dt = perf_counter() - t
x = model.sample(3000)
print(f"SGD iteration [for loop, only compile loss]\n\t{dt =:.3e} {dt/n_iter =:.3e} ")
print(x.mean(axis=0), jnp.cov(x, rowvar=False))

fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}(z)$", marker="*", s=5.0)
ax.scatter(*model._rngs.normal((x.shape[0], 2)).T, label=r"Target", s=3.0, marker="+")
ax.legend()

ax = axs[1]
ax.plot(fs, label=r"$\operatorname{KL}$")
ax1 = ax.twinx()
ax1.plot(grad_norms_sq, color="red", label=r"$\| \nabla L\|_2$")
ax1.set_yscale("log")
fig.legend()
fig.savefig("test_pm_euclidean.pdf")


def gd_step(_model, *args):
    (f, (x, logpdf)), grad = vg_fn(_model)
    gd, params, rest = nnx.split(_model, nnx.Param, ...)
    params_new = jax.tree.map(lambda x, y: x - y * lr, params, grad)
    nnx.update(_model, params_new)
    return _model, f


t = perf_counter()
model, fs = jax.lax.scan(gd_step, model, length=n_iter)
dt = perf_counter() - t
print(f"SGD iteration [lax scan, 1-st call]\n\t{dt =:.3e} {dt/n_iter =:.3e} ")

t = perf_counter()
model, fs = jax.lax.scan(gd_step, model, length=n_iter)
dt = perf_counter() - t
print(f"SGD iteration [lax scan, 2-nd call]\n\t{dt =:.3e} {dt/n_iter =:.3e} ")
