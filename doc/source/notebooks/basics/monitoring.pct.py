# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: 'Python 3.7.4 64-bit (''gpflow2'': conda)'
#     name: python37464bitgpflow2conda97d7f93b14fc49f496551bb0d0f2e18e
# ---

# %% [markdown]
# # Monitoring Optimisation
#
# In this notebook we cover how to monitor the model and certain metrics during optimisation.
#
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gpflow
from gpflow.ci_utils import ci_niter

# %% [markdown]
# The monitoring functionality lives in `gpflow.monitor`.
# For now, we import `ModelToTensorBoard`, `ImageToTensorBoard`, `ScalarToTensorBoard` monitoring tasks and `MonitorTaskGroup` and `Monitor`.

# %%
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

# %% [markdown]
# ## Set up data and model

# %%
# Define some configuration constants.

num_data = 100
noise_std = 0.1
optimisation_steps = ci_niter(100)

# %%
# Create dummy data.

X = np.random.randn(num_data, 1)  # [N, 2]
Y = np.sin(X) + 0.5 * np.cos(X) + np.random.randn(*X.shape) * noise_std  # [N, 1]
plt.plot(X, Y, "o")

# %%
# Set up model and print

kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 2.0]) + gpflow.kernels.Linear()
model = gpflow.models.GPR((X, Y), kernel, noise_variance=noise_std ** 2)
model


# %%
# We define a function that plots the model's prediction (in the form of samples) together with the data.
# Importantly, this function has no other argument than `fig: matplotlib.figure.Figure` and `axes: matplotlib.figure.Axes`.


def plot_prediction(fig, ax):
    Xnew = np.linspace(X.min() - 0.5, X.max() + 0.5, 100).reshape(-1, 1)
    Ypred = model.predict_f_samples(Xnew, full_cov=True, num_samples=20).numpy()
    ax.plot(Xnew.flatten(), np.squeeze(Ypred).T, "C1", alpha=0.2)
    ax.plot(X, Y, "o")


# Let's check if the function does the desired plotting
fig = plt.figure()
ax = fig.subplots()
plot_prediction(fig, ax)
plt.show()

# %% [markdown]
# ## Set up monitoring tasks
#
# We now define the `MonitorTask`s that will be executed during the optimisation.
# For this tutorial we set up three tasks:
# - `ModelToTensorBoard`: writes the models hyper-parameters such as `likelihood.variance` and `kernel.lengthscales` to a TensorBoard.
# - `ImageToTensorBoard`: writes custom matplotlib images to a TensorBoard.
# - `ScalarToTensorBoard`: writes any scalar value to a TensorBoard. Here, we use it to write the model's training objective.

# %%
log_dir = "logs"  # Directory where TensorBoard files will be written.
model_task = ModelToTensorBoard(log_dir, model)
image_task = ImageToTensorBoard(log_dir, plot_prediction, "image_samples")
lml_task = ScalarToTensorBoard(log_dir, lambda: model.log_likelihood(), "lml")

# %% [markdown]
# Finally, we collect all these tasks in a `MonitorCollection` object. This simple wrapper will call each task sequentially.

# %%
monitor = MonitorCollection([model_task, image_task, lml_task])


# %%
@tf.function
def closure():
    return -model.log_likelihood()


opt = tf.optimizers.Adam()

for step in range(optimisation_steps):
    opt.minimize(closure, model.trainable_variables)
    monitor(step)  # <-- run the monitoring

# %% [markdown]
# TensorBoard is accessible through the browser, after launching the server by running `tensorboard --logdir ${logdir}`. See the [TensorFlow documentation on TensorBoard](https://www.tensorflow.org/tensorboard/get_started) for more information.


# %% [markdown]
# For optimal performance, we can also wrap the monitor call inside `tf.function`:

# %%
opt = tf.optimizers.Adam()

model_task = ModelToTensorBoard(log_dir, model)
lml_task = ScalarToTensorBoard(log_dir, lambda: model.log_likelihood(), "lml")
# Note that the `ImageToTensorBoard` task cannot be compiled, and is omitted from the collection
monitor = MonitorCollection([model_task, lml_task])


# %% [markdown]
# In the optimisation loop below we use `tf.range` (rather than Python's built-in range) to avoid re-tracing the `step` function each time.


@tf.function
def step(i):
    opt.minimize(lambda: -1.0 * model.log_likelihood(), model.trainable_variables)
    monitor(i)


for i in tf.range(optimisation_steps):
    step(i)
