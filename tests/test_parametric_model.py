import os

# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


import sys
import time
import traceback
import argparse

import jax

jax.config.update("jax_debug_infs", True)
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import jax.scipy as jsp


from flax import nnx
import functools
from neuripp._ode._ode import *
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward

from time import perf_counter

import matplotlib.pyplot as plt

from test_rhs import LinearRHS, MLP
from operator import add


dim = 2
n_iter = 1_000
batch_size = 1000
lr = 1e-3
# rhs_net = LinearRHS(2, with_counter=False)
rhs_net = MLP(dim, dim_hidden=16, n_hidden=2)
model = ParametricPushforward(
    rhs_net,
    100,
    42,
    ode_nstep_max=10,
    divergence_method="hutchinson",
    ode_method="rk45",
    ode_kwargs=dict(h_max=0.5),
)

z = model._sample_latent(1000)
x = model(z)
zz = model.pullback(x)

fig, axs = plt.subplots(1, 3)

ax = axs[0]
ax.hist(jnp.abs(z - zz).ravel())

ax = axs[1]
ax.scatter(*z[:, :2].T, label="$z$", marker="+", s=1.0)
ax.scatter(*zz[:, :2].T, label=r"$T^{-1}(T(z))$", marker="*", s=1.0)
ax.scatter(*x[:, :2].T, label=r"T(z)", marker="o", s=1.4)
ax.legend()

x, logpdf = model.sample(1000, with_log_density=True)
ax = axs[2]
im = ax.scatter(*x[:, :2].T, c=logpdf, label=r"T(z)", marker="o", s=1.4)
ax.legend()

fig.colorbar(im, ax=axs[2])
fig.savefig("transport_init.pdf")

gd, params, rest = nnx.split(model, nnx.Param, ...)
params = jax.tree.map(lambda x: x * 1e-2, params)
nnx.update(model, params)


mean_target = jnp.ones(dim)
cov_target = jnp.diag(jnp.array([2.0, 0.5]))


#
# def logpdf_target(x):
#     return jsp.special.logsumexp(
#         jnp.stack(
#             (
#                 jsp.stats.multivariate_normal.logpdf(
#                     x, mean=-mean_target, cov=jnp.eye(dim) / 9.0
#                 ),
#                 jsp.stats.multivariate_normal.logpdf(
#                     x, mean=mean_target, cov=jnp.eye(dim) / 9.0
#                 ),
#             ),
#             axis=-1,
#         ),
#         axis=-1,
#     )
#
def logpdf_target(x):
    return jsp.stats.multivariate_normal.logpdf(x, mean_target, cov_target)


def loss(pm: ParametricPushforward):
    x, logpdf = pm.sample(batch_size, with_log_density=True)
    loss_val = (logpdf - logpdf_target(x)).mean()
    return loss_val, (x, logpdf)


vg_fn = nnx.jit(nnx.value_and_grad(loss, has_aux=True))

t = perf_counter()
(f, (x, logpdf)), grad = vg_fn(model)
dt = perf_counter() - t

print(f"First compute of loss \n\t{dt =:.3e}")

fs = []
grad_norms_sq = []


t = perf_counter()
for k in range(n_iter):
    jax.debug.print("k = {k}", k=k)
    try:
        (f, (x, logpdf)), grad = vg_fn(model)
    except:
        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        ax.scatter(*x[:, :2].T, label=r"Current model", marker="*", s=5.0)
        ax = axs[1]
        ax.plot(fs, label="Loss")
        ax1 = ax.twinx()
        ax1.plot(grad_norms_sq, color="red", label=r"$\| \nabla L\|_2$")

        fig.savefig("debug.pdf")

        raise

    gd, params, rest = nnx.split(model, nnx.Param, ...)
    params_new = jax.tree.map(lambda x, y: x - y * lr, params, grad)
    nnx.update(model, params_new)
    fs.append(f)

    grad_norm_sq = jax.tree.reduce_associative(
        add, jax.tree.map(lambda _x: (_x**2).ravel().sum(), grad)
    )
    grad_norms_sq.append(grad_norm_sq)

fs = jnp.array(fs)
dt = perf_counter() - t
print(x.mean(axis=0), jnp.cov(x, rowvar=False))

print(fs)

print(f"SGD iteration [for loop]\n\t{dt =:.3e} {dt/n_iter =:.3e} ")


x = model.sample(3000)
fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.scatter(*x[:, :2].T, label=r"$T_{\text{opt}}(z)$", marker="*", s=5.0)
ax.scatter(*model._rngs.normal((x.shape[0], 2)).T, label=r"Target", s=3.0, marker="+")
ax.legend()

ax = axs[1]
ax.plot(fs)
fig.savefig("test_pm.pdf")


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
