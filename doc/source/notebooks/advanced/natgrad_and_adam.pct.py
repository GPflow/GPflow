# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook compares performance and time for a SVGP model trained on a simple 1-D periodic dataset for 3 optimizers:
#
# 1. Adam only
# 2. Adam and Natgrad separately
# 3. (New optimizer) Adam+NatGrad combined
#
# Results show that 2) and 3) achieve better ELBO than 1). Additionally, 1) takes ~20s, 2) takes ~30s, and 3) takes ~23s for 20k iterations on this dataset.

# %%
# %matplotlib inline
import itertools
import numpy as np
import time
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt

from gpflow.optimizers import JointNaturalGradientAndAdam

plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)


# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)


N = 10000  # Number of training observations

X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")


# %%
def get_model_and_loss(minibatch_size=100, M=50):
    """
    :param M: number of inducing point locations
    """
    Z = X[:M, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

    kernel = gpflow.kernels.SquaredExponential()
    model = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X, Y))
        .repeat()
        .shuffle(N)
        .batch(minibatch_size, drop_remainder=True)
    )
    training_loss = model.training_loss_closure(iter(train_dataset), compile=True)

    return model, training_loss


# %% [markdown]
# # ADAM

# %%
num_iterations = 20000

optimizer = tf.optimizers.Adam()

model, training_loss = get_model_and_loss()


@tf.function
def optimization_step():
    optimizer.minimize(training_loss, model.trainable_variables)


logf = []
# Time the training loop
d = time.time()

for step in range(num_iterations):
    optimization_step()
    if step % 10 == 0:
        elbo = -training_loss().numpy()
        logf.append(elbo)

print(f"Time taken {time.time() - d}s.")

plt.plot(np.arange(num_iterations)[::10], logf)
plt.xlabel("iteration")
_ = plt.ylabel("ELBO")

# %% [markdown]
# # ADAM, NatGrad separately

# %%
num_iterations = 20000

adam_opt = tf.optimizers.Adam()
natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)

model, training_loss = get_model_and_loss()

# Get variational/nonvariational params.
# Run Adam on nonvariational, NatGrad on variational.
"""
TODO: Code below fails to work. Make it work.
variational_params = [(model.q_mu, model.q_sqrt)]
non_variational_params = [p for p in model.trainable_parameters if p not in variational_params
"""
variational_params = [(model.q_mu, model.q_sqrt)]
[gpflow.set_trainable(vp, False) for vp in variational_params]
non_variational_params = model.trainable_variables


@tf.function
def optimization_step():
    # TODO: this rolls the data because of iterator?!?!
    adam_opt.minimize(training_loss, non_variational_params)
    natgrad_opt.minimize(training_loss, variational_params)


logf = []
# Time the training loop
d = time.time()

for step in range(num_iterations):
    optimization_step()
    if step % 10 == 0:
        elbo = -training_loss().numpy()
        logf.append(elbo)

print(f"Time taken {time.time() - d}s.")

plt.plot(np.arange(num_iterations)[::10], logf)
plt.xlabel("iteration")
_ = plt.ylabel("ELBO")

# %% [markdown]
# # ADAM + NatGrad combined

# %%
num_iterations = 20000

optimizer = JointNaturalGradientAndAdam(gamma=0.1, adam_lr=0.001)

model, training_loss = get_model_and_loss()

# Get variational/nonvariational params.
variational_params = [(model.q_mu, model.q_sqrt)]
[gpflow.set_trainable(vp, False) for vp in variational_params]
non_variational_params = model.trainable_variables


@tf.function
def optimization_step():
    optimizer.minimize(training_loss, variational_params, non_variational_params)


logf = []

# Time the training loop
d = time.time()

for step in range(num_iterations):
    optimization_step()
    if step % 10 == 0:
        elbo = -training_loss().numpy()
        logf.append(elbo)

print(f"Time taken {time.time() - d}s.")

plt.plot(np.arange(num_iterations)[::10], logf)
plt.xlabel("iteration")
_ = plt.ylabel("ELBO")

# %%
