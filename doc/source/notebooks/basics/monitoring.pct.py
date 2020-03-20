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
# The monitoring functionality lives in `gpflow.utilities.monitor`. For now, we import `ModelToTensorBoardTask`, `ImageToTensorBoardTask`, `ScalarToTensorBoardTask` monitoring tasks and `TasksCollection`.

# %%
from gpflow.monitor import (
    ModelToTensorBoardTask,
    ImageToTensorBoardTask,
    ScalarToTensorBoardTask,
    TasksCollection,
)

# %% [markdown]
# ## Setup data and model

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
# Setup model and print

kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 2.0]) + gpflow.kernels.Linear()
model = gpflow.models.GPR((X, Y), kernel, noise_variance=noise_std ** 2)
model


# %%
# We define a function that plot's the model's prediction (in the form of samples) togheter with the data.
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
# ## Setup monitoring tasks
#
# We now define the `gpflow.utilities.monitor.MonitorTask`s that will be executed during the optimisation.
# For this tutorial we set up three tasks:
# - `ModelToTensorBoardTask`: writes the models hyper-parameters such as `likelihood.variance` and `kernel.lengthscales` to a TensorBoard.
# - `ImageToTensorBoardTask`: writes custom matplotlib images to a TensorBoard.
# - `ScalarToTensorBoardTask`: writes any scalar value to a TensorBoard. Here, we use it to write the model's `objective`.

# %%
log_dir = "logs"  # Directory where TensorBoard files will be written.
model_task = ModelToTensorBoardTask(log_dir, model)
image_task = ImageToTensorBoardTask(log_dir, plot_prediction, "image_samples")
lml_task = ScalarToTensorBoardTask(log_dir, lambda: model.log_likelihood().numpy(), "lml")

# %% [markdown]
# Finally, we collect all these tasks in a `TasksCollection` object. This simple wrapper will call each task sequentially.

# %%
monitor_tasks = TasksCollection([model_task, image_task, lml_task])


# %%
@tf.function
def closure():
    return -model.log_likelihood()


opt = tf.optimizers.Adam()

for step in range(optimisation_steps):
    opt.minimize(closure, model.trainable_variables)
    monitor_tasks(step)  # <-- run the monitoring

# %% [markdown]
# TensorBoard is accessable through the browser, after lauching the servers `tensorboard --logdir ${logdir}`. See [TF docs](https://www.tensorflow.org/tensorboard/get_started) for more info.
