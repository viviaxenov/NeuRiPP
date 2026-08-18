import jax
import jax.numpy as jnp
from flax import nnx
from datasets import load_dataset

from neuripp.parametric_pushforward.parametric_pushforward import ParametricPushforward
from neuripp.methods.ngd import get_ngd
from neuripp.functionals import CrossEntropy
from rhs_architectures import *
from data_generators import DatasetBatcher

import matplotlib.pyplot as plt


rngs = nnx.Rngs(42)
rhs = CFMConv2D((28, 28), rngs, n_channels=64)

model = ParametricPushforward(rhs, rngs, 150, ode_nstep_max=5, ode_method="euler")
print(model)

ds = load_dataset("zalando-datasets/fashion_mnist")

X_train = jnp.array(ds["train"]["image"], dtype=jnp.float32) / 256.0
X_test = jnp.array(ds["test"]["image"], dtype=jnp.float32) / 256.0

mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean[None, ...]) / std[None, ...]
X_test = (X_test - mean[None, ...]) / std[None, ...]

X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

batcher = DatasetBatcher(100, 10, X_train)
loss = CrossEntropy.cross_entropy

init, step = get_ngd(loss, natural_grad_clipping_threshold=0.1)

state = init(model, args=(0.001,), kwargs=dict(linear_solver_regularization=0.01))

for _ in range(20):
    batch = batcher(rngs)
    state, vals = step(state, batch, rngs)
    print(f"loss = {vals[0]:.2e}")

model = state[0]

X_m = model.sample(9, rngs)

images = (X_m*std + mean).reshape(-1, 28,28)

fig, axs = plt.subplots(3,3)
axs = axs.ravel()

for ax, im in zip(axs, images):
    ax.imshow(images)
    ax.axis_off()

plt.show()
