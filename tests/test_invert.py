import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
from neuripp.methods.ngd import get_ngd
from neuripp.methods.sgd import get_sgd
from neuripp.utility.utility import *

from time import perf_counter

import matplotlib.pyplot as plt

from test_rhs import LinearRHS, MLP

import tqdm

dim = 2

N_mc = 100
batch_size = 512

rhs_net = MLP(dim, dim_hidden=128, n_hidden=2, activation=nnx.swish)
model_args = (
    rhs_net,
    N_mc,
    42,
)
model_kwargs = dict(
    ode_nstep_max=25,
    divergence_method="hutchinson",
    # ode_method="euler",
    ode_method="rk45",
    ode_kwargs=dict(h_max=0.1, N_iter_to_accept=15, adaptive=True)
)

data_gen = checkerboard_generator(batch_size, 30)

model = ParametricPushforward(*model_args, **model_kwargs)

z = model._sample_latent(512)
x, logpdf = model.pushforward(z, with_log_density=True)
zz, logpdf_inv = model.pullback(x, with_log_density=True)

diff_sol = jnp.linalg.norm(z - zz, axis=-1)
diff_logpdf = jnp.abs(logpdf - logpdf_inv)

fig, axs = plt.subplots(nrows=1, ncols=3, layout='constrained')

axs[0].hist(diff_sol, color='blue', label=r'$\|z - T^{-1}(T(z))\|_2$')
axs[1].hist(diff_logpdf, color='orange', label=r'$|\log\rho_f - \log\rho_i|$')
axs[2].scatter(*z[:, :2].T, marker='1', label=r'$z$')
axs[2].scatter(*zz[:, :2].T, marker='2', label=r'$T^{-1}(T(z))$')

fig.legend(loc='outside lower center', ncols=2)
fig.savefig("test_inv_z.pdf")


x = next(data_gen)
z, logpdf_inv = model.pullback(x, with_log_density=True)
xx, logpdf = model.pushforward(z, with_log_density=True)

diff_sol = jnp.linalg.norm(x - xx, axis=-1)
diff_logpdf = jnp.abs(logpdf - logpdf_inv)

fig, axs = plt.subplots(nrows=1, ncols=3, layout='constrained')

axs[0].hist(diff_sol, color='blue', label=r'$\|z - T^{-1}(T(z))\|_2$')
axs[1].hist(diff_logpdf, color='orange', label=r'$|\log\rho_f - \log\rho_i|$')
axs[2].scatter(*x[:, :2].T, marker='1', label=r'Data')
axs[2].scatter(*xx[:, :2].T, marker='2', label=r'$T(T^{-1}(x))$')

fig.legend(loc='outside lower center', ncols=2)
fig.savefig("test_inv_x.pdf")

