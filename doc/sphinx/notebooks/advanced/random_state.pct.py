# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Managing random state
#
# Many GPflow methods take an optional `seed` parameter. This can take three types of values:
# * `None`.
# * An integer.
# * A tensor of shape `[2]` and `dtype=tf.int32`.
#
# Below we will go over how these are interpreted, and what that means.

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow

# %% [markdown]
# Let us quickly create a model to test on:

# %%
model = gpflow.models.GPR(
    (np.zeros((0, 1)), np.zeros((0, 1))),
    kernel=gpflow.kernels.SquaredExponential(),
)
Xplot = np.linspace(0.0, 10.0, 100)[:, None]

# %% [markdown]
# ## Seed `None`
#
# When the seed is set to `None`, the randomness depends on the state of the TensorFlow global seed. This is the default behaviour.

# %%
tf.random.set_seed(123)
Yplot = model.predict_f_samples(Xplot, seed=None)
plt.plot(Xplot, Yplot)

tf.random.set_seed(123)
Yplot = model.predict_f_samples(Xplot, seed=None)
plt.plot(Xplot, Yplot, ls=":")

tf.random.set_seed(456)
Yplot = model.predict_f_samples(Xplot, seed=None)
_ = plt.plot(Xplot, Yplot, ls="-.")

# %% [markdown]
# ## Integer seed
# When the seed is set to an integer, the randomness depends on both the TensorFlow global seed, and the `seed` passed to the method.

# %%
tf.random.set_seed(123)
Yplot = model.predict_f_samples(Xplot, seed=1)
plt.plot(Xplot, Yplot)

tf.random.set_seed(123)
Yplot = model.predict_f_samples(Xplot, seed=1)
plt.plot(Xplot, Yplot, ls=":")

tf.random.set_seed(123)
Yplot = model.predict_f_samples(Xplot, seed=2)
plt.plot(Xplot, Yplot, ls="-.")

tf.random.set_seed(456)
Yplot = model.predict_f_samples(Xplot, seed=2)
_ = plt.plot(Xplot, Yplot, ls="--")

# %% [markdown]
# ## Full random state as seed
# If you set the `seed` to a tensor of shape `[2]` and `dtype=tf.int32`, that will completely define the randomness used, and the TensorFlow global seed will be ignored.

# %%
tf.random.set_seed(123)
Yplot = model.predict_f_samples(
    Xplot, seed=tf.constant([12, 34], dtype=tf.int32)
)
plt.plot(Xplot, Yplot)

tf.random.set_seed(456)
Yplot = model.predict_f_samples(
    Xplot, seed=tf.constant([12, 34], dtype=tf.int32)
)
plt.plot(Xplot, Yplot, ls=":")

tf.random.set_seed(456)
Yplot = model.predict_f_samples(
    Xplot, seed=tf.constant([56, 78], dtype=tf.int32)
)
_ = plt.plot(Xplot, Yplot, ls="-.")


# %% [markdown]
# When using the full state as random seed it is important you are careful about when you do and do not reuse the seed. If you write a function, that takes a seed, and that function makes multiple calls, that also takes seeds, you should generally be careful to pass different seeds to the called functions. You can use [tfp.random.split_seed](https://www.tensorflow.org/probability/api_docs/python/tfp/random/split_seed) to create multiple new seeds from one seed. For example:

# %%
def calls_two(seed: tf.Tensor) -> None:
    s1, s2 = tfp.random.split_seed(seed)
    model.predict_f_samples(Xplot, seed=s1)
    model.predict_f_samples(Xplot, seed=s2)


def calls_in_loop(seed: tf.Tensor) -> None:
    for _ in range(5):
        seed, s = tfp.random.split_seed(seed)
        calls_two(s)


calls_in_loop(tf.constant([12, 34], dtype=tf.int32))

# %% [markdown]
# ## Stable randomness in model optimisation
#
# By default the `training_loss` method has `seed = None`, which means that every call to it will have different randomness. Some of the GPflow likelihoods rely on randomness, and this means your loss function will become noisy. You can get a deterministic loss function by using `training_loss_closure` instead, which allows you to bind a fixed seed to your loss:

# %%
X = np.array([[1.0], [3.0], [9.0]])
Y = np.array([[0.0], [2.0], [2.0]])
model = gpflow.models.GPR(
    (X, Y),
    kernel=gpflow.kernels.SquaredExponential(),
)
fixed_seed = tf.constant([12, 34], dtype=tf.int32)
opt = gpflow.optimizers.Scipy()
opt.minimize(
    model.training_loss_closure(seed=fixed_seed), model.trainable_variables
)


# %% [markdown]
# ## Stable randomness used in optimisation
#
# Generally random sampling with a fixed random state is stable enough to be used over changing model parameters. For example, observe how adding data to a model warps the random sample:

# %%
model1 = gpflow.models.GPR(
    (np.zeros((0, 1)), np.zeros((0, 1))),
    kernel=gpflow.kernels.SquaredExponential(),
)

X = np.array([[1.0], [3.0], [9.0]])
Y = np.array([[0.0], [2.0], [2.0]])

model2 = gpflow.models.GPR(
    (X, Y),
    kernel=gpflow.kernels.SquaredExponential(),
)

fixed_seed = tf.constant([12, 34], dtype=tf.int32)

plt.scatter(X, Y, c="C1")

Yplot = model1.predict_f_samples(Xplot, seed=fixed_seed)
plt.plot(Xplot, Yplot)

Yplot = model2.predict_f_samples(Xplot, seed=fixed_seed)
_ = plt.plot(Xplot, Yplot)

# %% [markdown]
# Or, let us try optimising model parameters to fit a random sample to data:

# %%
model = gpflow.models.GPR(
    (np.zeros((0, 1)), np.zeros((0, 1))),
    kernel=gpflow.kernels.SquaredExponential(),
)

X = np.array([[1.0], [3.0], [9.0]])
Y = np.array([[0.0], [-1.0], [2.0]])
fixed_seed = tf.constant([12, 34], dtype=tf.int32)

plt.scatter(X, Y, c="black")

Yplot = model.predict_f_samples(Xplot, seed=fixed_seed)
plt.plot(Xplot, Yplot)

data_indices = tf.searchsorted(Xplot[:, 0], X[:, 0])


def loss() -> tf.Tensor:
    Yplot = model.predict_f_samples(Xplot, seed=fixed_seed)
    Ypred = tf.gather(Yplot, data_indices)
    delta = Ypred - Y
    return tf.reduce_sum(delta ** 2)


opt = gpflow.optimizers.Scipy()
opt.minimize(loss, model.trainable_variables)

Yplot = model.predict_f_samples(Xplot, seed=fixed_seed)
_ = plt.plot(Xplot, Yplot)

# %% [markdown]
# ## Implementation details
#
# Behind the scenes any `seed` is pushed through [tfp.random.sanitize_seed](https://www.tensorflow.org/probability/api_docs/python/tfp/random/sanitize_seed), and fed to `tf.random.stateless_*`.
