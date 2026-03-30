import sys
import time
import traceback
import argparse
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax._src import ad_util
from flax import nnx
import functools
from neuripp._ode._ode import *
from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
import matplotlib.pyplot as plt

from test_rhs import LinearRHS, MLP


dim = 2
# rhs_net = LinearRHS(2, with_counter=False)
rhs_net = MLP(dim, dim_hidden=16)
model = ParametricPushforward(
    rhs_net, 100, 42, ode_nstep_max=10, divergence_method="hutchinson"
)

z = model._sample_latent(1000)
x = model(z)
zz = model.pullback(x)

fig, axs = plt.subplots(1, 2)

ax = axs[0]
ax.hist(jnp.abs(z - zz).ravel())

ax = axs[1]
ax.scatter(*z[:, :2].T, label="$z$", marker="+", s=1.0)
ax.scatter(*zz[:, :2].T, label=r"$T^{-1}(T(z))$", marker="*", s=1.0)
ax.scatter(*x[:, :2].T, label=r"T(z)", marker="o", s=1.4)
ax.legend()

gd, params, rest = nnx.split(model, nnx.Param, ...)
params = jax.tree.map(lambda x: x * 1e-2, params)
nnx.update(model, params)

fig, axs = plt.subplots(1, 2)
ax = axs[0]
x, logpdf = model.sample(1000, with_log_density=True)
ax.scatter(*x[:, :2].T, label=r"T_0(z)", marker="o", s=3.5)



def loss(pm: ParametricPushforward):
    x, logpdf = pm.sample(1_000, with_log_density=True)
    loss_val = (
        jsp.stats.multivariate_normal.logpdf(x, mean=jnp.ones(dim), cov=jnp.eye(dim))
        - logpdf
    ).mean()
    # x = pm.sample(2, with_log_density=False)
    # loss_val = -x.mean()
    return loss_val, (x, logpdf)


vg_fn = nnx.jit(nnx.value_and_grad(loss, has_aux=True))
fs = []

for _ in range(1000):
    (f, (x, logpdf)), grad = vg_fn(model)
    gd, params, rest = nnx.split(model, nnx.Param, ...)
    params_new = jax.tree.map(lambda x, y: x - y * 6e-4, params, grad)
    nnx.update(model, params_new)
    fs.append(f)


ax.scatter(*x[:, :2].T, label=r"T_{\text{opt}}(z)", marker="*", s=5.0)
ax.scatter(*model._rngs.normal((x.shape[0], 2)).T, label=r"Target", s=3., marker='+')
ax.legend()

ax = axs[1]
ax.plot(fs)

plt.show()
print(x)
