import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import jax
import jax.numpy as jnp
from flax import nnx
from datasets import load_dataset

from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.methods.ngd import get_ngd
from neuripp.methods.optax_optimizer import *
from neuripp.functionals import CrossEntropy
from rhs_architectures import *
from data_generators import DatasetBatcher

import matplotlib.pyplot as plt

from tqdm import trange

from jax.ad_checkpoint import print_saved_residuals


def normalize(X, moments=None):
    if moments is None:
        moments = X.mean(), X.std()
    mean, std = moments
    return (X - mean[jnp.newaxis, ...]) / jnp.maximum(std, 1e-20)[
        jnp.newaxis, ...
    ], moments


dataset = sys.argv[1] if len(sys.argv) > 1 else "mnist"
method = sys.argv[2] if len(sys.argv) > 2 else "ngd"

match dataset:
    case "mnist":
        ds = load_dataset("ylecun/mnist")
    case "fashion":
        ds = load_dataset("zalando-datasets/fashion_mnist")
    case _:
        raise ValueError(f"{dataset=} not supported")

batch_size = 256
N_mc = batch_size


rngs = nnx.Rngs(42)
rhs = CFMConv2D((28, 28), rngs, n_channels=64)

model = ParametricPushforward(
    rhs,
    rngs,
    N_mc,
    ode_nstep_max=12,
    ode_method="rk45",
    ode_kwargs=dict(
        h_max=0.3,
        adaptive=True,
        grad_checkpointing="dots_only",
    ),
)


X_train = jnp.array(ds["train"]["image"], dtype=jnp.float32) / 255.0
X_test = jnp.array(ds["test"]["image"], dtype=jnp.float32) / 255.0

X_train, moments = normalize(X_train)
X_test, _ = normalize(X_test, moments)

X_train = X_train.reshape((-1, 28 * 28))
X_test = X_test.reshape((-1, 28 * 28))

batcher = DatasetBatcher(batch_size, 1, X_train)
loss = CrossEntropy.cross_entropy

model.train()

match method:
    case "ngd":
        init, step = get_ngd(loss, natural_grad_clipping_threshold=1.)
        state = init(
            model, args=(0.001,), kwargs=dict(linear_solver_regularization=0.01)
        )
    case x if x in optax_optimizers:
        init, step = get_optax(loss, method=x)
        state = init(model, optimizer_args=(0.001,), optimizer_kwargs=dict())

step = nnx.jit(step)

pbar = trange(120)
loss_min = jnp.inf

for _ in pbar:
    batch = batcher(rngs)
    state, vals = step(state, batch, rngs)
    loss_val = vals[0]
    loss_min = jnp.minimum(loss_val, loss_min)
    pbar.set_description(f"loss = {loss_val:.3f} {loss_min=:.3f}")

model = state[0]
model.eval()

npic_x, npic_y = (6, 6)

X_m = model.sample(npic_x * npic_y, rngs)

mean, std = moments
images = X_m.reshape(-1, 28, 28) * std + mean

fig, axs = plt.subplots(npic_x, npic_y, layout="constrained", figsize=(30, 30))
axs = axs.ravel()

for ax, im in zip(axs, images):
    ax.imshow(im)
    ax.set_axis_off()

fig.savefig(f"{dataset}_samples_{method}.pdf")
