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
# In this notebook we cover how to monitor the metrics during optimisation.

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gpflow
from gpflow.utilities.monitor import ModelToTensorBoardTask, ImageToTensorBoardTask, ScalarToTensorBoard, TasksCollection
from gpflow.ci_utils import ci_niter

# %%
num_data = 100
noise_std = 0.1
optimisation_steps = ci_niter(100)
log_dir = "logs"

# %%
X = np.random.randn(num_data, 1)  # [N, 2]
Y = np.sin(X) + 0.5 * np.cos(X) + np.random.randn(*X.shape) * noise_std  # [N, 1]
plt.plot(X, Y, "o")

# %%
kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 2.0]) + gpflow.kernels.Linear()
model = gpflow.models.GPR((X, Y), kernel, noise_variance=noise_std ** 2)
model


# %%
def plot_prediction(fig, ax):
    Xnew = np.linspace(X.min() - 0.5, X.max() + 0.5, 100).reshape(-1, 1)
    Ypred = model.predict_f_samples(Xnew, full_cov=True, num_samples=20).numpy()
    ax.plot(Xnew.flatten(), np.squeeze(Ypred).T, "C1", alpha=.2)
    ax.plot(X, Y, "o")

fig = plt.figure()
ax = fig.subplots()
plot_prediction(fig, ax)
plt.show()

# %%
model_task = ModelToTensorBoardTask(log_dir, model)
image_task = ImageToTensorBoardTask(log_dir, plot_prediction, "image_samples")
lml_task = ScalarToTensorBoard(log_dir, lambda: model.log_likelihood().numpy(), "lml")
monitor_tasks = TasksCollection([model_task, image_task, lml_task])


# %%
@tf.function
def closure():
    return - model.log_likelihood()

opt = tf.optimizers.Adam()

for step in range(optimisation_steps):
    opt.minimize(closure, model.trainable_variables)
    monitor_tasks(step)
