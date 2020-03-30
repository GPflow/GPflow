# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# GPflow with TensorFlow 2
# ===
#
# ##### Small steps big changes
#
# <br>
#
#

# %%
from typing import Tuple, Optional
import tempfile
import pathlib

import datetime
import io
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.config import default_float
from gpflow.utilities import to_default_float

import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# Make `tensorboard` work inside notebook:

# %%
output_logdir = "/tmp/tensorboard"

# !rm -rf "{output_logdir}"
# !mkdir "{output_logdir}"

# %load_ext tensorboard
# %matplotlib inline


def enumerated_logdir(_logdir_id: int = [0]):
    logdir = pathlib.Path(output_logdir, str(_logdir_id[0]))
    _logdir_id[0] += 1
    return str(logdir)


# %% [markdown]
# Set up random seeds and default float for `gpflow` tensors:

# %%
gpflow.config.set_default_float(np.float64)
np.random.seed(0)
tf.random.set_seed(0)


# %% [markdown]
# ## Loading data using TensorFlow Datasets
#
# For this example, we create a synthetic dataset (noisy sine function):

# %%
def noisy_sin(x):
    return tf.math.sin(x) + 0.1 * tf.random.normal(x.shape, dtype=default_float())


num_train_data, num_test_data = 100, 500

X = tf.random.uniform((num_train_data, 1), dtype=default_float()) * 10
Xtest = tf.random.uniform((num_test_data, 1), dtype=default_float()) * 10

Y = noisy_sin(X)
Ytest = noisy_sin(Xtest)

data = (X, Y)

plt.plot(X, Y, "xk")
plt.show()

# %% [markdown]
# Working with TensorFlow Datasets is an efficient way to rapidly shuffle, iterate, and batch from data.

# %%
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
test_dataset = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))

batch_size = 32
num_features = 10
prefetch_size = num_train_data // 2
shuffle_buffer_size = num_train_data // 2
num_batches_per_epoch = num_train_data // batch_size

original_train_dataset = train_dataset
train_dataset = (
    train_dataset.repeat()
    .prefetch(prefetch_size)
    .shuffle(buffer_size=shuffle_buffer_size)
    .batch(batch_size)
)

print(f"prefetch_size={prefetch_size}")
print(f"shuffle_buffer_size={shuffle_buffer_size}")
print(f"num_batches_per_epoch={num_batches_per_epoch}")

# %% [markdown]
# ## Define a GP model
#
# In GPflow 2.0, we use `tf.Module` (or the very thin `gpflow.base.Module` wrapper) to build all our models, as well as their components (kernels, likelihoods, parameters, and so on).

# %%
kernel = gpflow.kernels.SquaredExponential(variance=2.0)
likelihood = gpflow.likelihoods.Gaussian()
inducing_variable = np.linspace(0, 10, num_features).reshape(-1, 1)

model = gpflow.models.SVGP(
    kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable
)

# %% [markdown]
# You can set a module (or a particular parameter) to be non-trainable using the auxiliary method ```set_trainable(module, False)```:

# %%
from gpflow import set_trainable

set_trainable(likelihood, False)
set_trainable(kernel.variance, False)

set_trainable(likelihood, True)
set_trainable(kernel.variance, True)

# %% [markdown]
# We can use ```param.assign(value)``` to assign a value to a parameter:

# %%
kernel.lengthscales.assign(0.5)

# %% [markdown]
# All these changes are reflected when we use ```print_summary(model)``` to print a detailed summary of the model. By default the output is displayed in a minimalistic and simple table.

# %%
from gpflow.utilities import print_summary

print_summary(model)  # same as print_summary(model, fmt="simple")

# %% [markdown]
# We can change default printing so that it will look nicer in our notebook:

# %%
gpflow.config.set_default_summary_fmt("notebook")

print_summary(model)  # same as print_summary(model, fmt="notebook")

# %% [markdown]
# Jupyter notebooks also format GPflow classes (that are subclasses of `gpflow.base.Module`) in the same nice way when at the end of a cell (this is independent of the `default_summary_fmt`):

# %%
model

# %% [markdown]
# ## Training using Gradient Tapes
#
# In TensorFlow 2, we can optimize (trainable) model parameters with TensorFlow optimizers using `tf.GradientTape`. In this simple example, we perform one gradient update of the Adam optimizer to minimize the negative marginal log likelihood (or ELBO) of our model.

# %%
optimizer = tf.optimizers.Adam()

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(model.trainable_variables)
    obj = -model.elbo(data)
grads = tape.gradient(obj, model.trainable_variables)

optimizer.apply_gradients(zip(grads, model.trainable_variables))


# %% [markdown]
# For a more elaborate example of a gradient update we can define an ```optimization_step``` that uses the decorator ```tf.function``` on a closure. A closure is a callable that returns the model objective evaluated at a given dataset when called.

# %%
def optimization_step(model: gpflow.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor]):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        obj = -model.elbo(batch)
    grads = tape.gradient(obj, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# %% [markdown]
# We can use the functionality of TensorFlow Datasets to define a simple training loop that iterates over batches of the training dataset:

# %%
def simple_training_loop(model: gpflow.models.SVGP, epochs: int = 1, logging_epoch_freq: int = 10):
    batches = iter(train_dataset)
    tf_optimization_step = tf.function(optimization_step)
    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            tf_optimization_step(model, next(batches))

        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {model.elbo(data)}")


# %%
simple_training_loop(model, epochs=10, logging_epoch_freq=2)

# %% [markdown]
# ## Monitoring
#
# `gpflow.monitor` provides a thin wrapper on top of tf.summary that makes it easy to monitor the training procedure.
# For a more detailed tutorial see the [monitoring notebook](./basics/monitoring.pct.py).

# %%
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    ExecuteCallback,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

samples_input = np.linspace(0, 10, 100).reshape(-1, 1)


def plot_model(fig, ax):
    tf.print("Plotting...")
    mean, var = model.predict_f(samples_input)
    num_samples = 10
    samples = model.predict_f_samples(samples_input, num_samples)
    ax.plot(samples_input, mean, "C0", lw=2)
    ax.fill_between(
        samples_input[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )
    ax.plot(X, Y, "kx")
    ax.plot(samples_input, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    ax.set_ylim(-2.0, +2.0)
    ax.set_xlim(0, 10)


def print_cb(epoch_id=None, data=None):
    tf.print(f"Epoch {epoch_id}: ELBO (train)", model.elbo(data))


def elbo_cb(data=None, **_):
    return model.elbo(data)


output_logdir = enumerated_logdir()

model_task = ModelToTensorBoard(output_logdir, model)
elbo_task = ScalarToTensorBoard(output_logdir, elbo_cb, "elbo")
print_task = ExecuteCallback(callback=print_cb)

# We group these tasks and specify a period of `100` steps for them
fast_tasks = MonitorTaskGroup([model_task, elbo_task, print_task], period=100)

# We also want to see the model's fit during the optimisation
image_task = ImageToTensorBoard(output_logdir, plot_model, "samples_image")

# We typically don't want to plot too frequently during optimisation,
# which is why we specify a larger period for this task.
slow_taks = MonitorTaskGroup(image_task, period=500)
monitor = Monitor(fast_tasks, slow_taks)


def monitored_training_loop(epochs: int):
    tf_optimization_step = tf.function(optimization_step)
    batches = iter(train_dataset)

    for epoch in range(epochs):
        for _ in range(num_batches_per_epoch):
            batch = next(batches)
            tf_optimization_step(model, batch)

        epoch_id = epoch + 1
        monitor(epoch, epoch_id=epoch_id, data=data)


# %% [markdown]
# NOTE: for optimal performance it is recommended to wrap the monitoring inside `tf.function`.
# This is detailed in the [monitoring notebook](./basics/monitoring.ipynb).


# %%
model = gpflow.models.SVGP(
    kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable
)

monitored_training_loop(epochs=1000)

# %% [markdown]
# Then, we can use TensorBoard to examine the training procedure in more detail

# %%
# # %tensorboard --logdir "{output_logdir}"

# %% [markdown]
# ## Saving and loading models
#
# ### Checkpointing
#
# With the help of `tf.train.CheckpointManager` and `tf.train.Checkpoint`, we can checkpoint the model throughout the training procedure. Let's start with a simple example using checkpointing to save and load a `tf.Variable`:

# %%
initial_value = 1.2
a = tf.Variable(initial_value)

# %% [markdown]
# Create `Checkpoint` object:

# %%
ckpt = tf.train.Checkpoint(a=a)
manager = tf.train.CheckpointManager(ckpt, output_logdir, max_to_keep=3)

# %% [markdown]
# Save the variable `a` and change its value right after:

# %%
manager.save()
_ = a.assign(0.33)

# %% [markdown]
# Now we can restore the old variable value:

# %%
print(f"Current value of variable a: {a.numpy():0.3f}")

ckpt.restore(manager.latest_checkpoint)

print(f"Value of variable a after restore: {a.numpy():0.3f}")

# %% [markdown]
# In the example below, we modify a simple training loop to save the model every 100 epochs using the `CheckpointManager`.

# %%
model = gpflow.models.SVGP(
    kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable
)


def checkpointing_training_loop(
    model: gpflow.models.SVGP,
    batch_size: int,
    epochs: int,
    manager: tf.train.CheckpointManager,
    logging_epoch_freq: int = 100,
    epoch_var: Optional[tf.Variable] = None,
    step_var: Optional[tf.Variable] = None,
):
    tf_optimization_step = tf.function(optimization_step)
    batches = iter(train_dataset)

    for epoch in range(epochs):
        for step in range(num_batches_per_epoch):
            tf_optimization_step(model, next(batches))
            if step_var is not None:
                step_var.assign(epoch * num_batches_per_epoch + step + 1)
        if epoch_var is not None:
            epoch_var.assign(epoch + 1)

        epoch_id = epoch + 1
        if epoch_id % logging_epoch_freq == 0:
            ckpt_path = manager.save()
            tf.print(f"Epoch {epoch_id}: ELBO (train) {model.elbo(data)}, saved at {ckpt_path}")


# %%
step_var = tf.Variable(1, dtype=tf.int32, trainable=False)
epoch_var = tf.Variable(1, dtype=tf.int32, trainable=False)
ckpt = tf.train.Checkpoint(model=model, step=step_var, epoch=epoch_var)
manager = tf.train.CheckpointManager(ckpt, output_logdir, max_to_keep=5)

print(f"Checkpoint folder path at: {output_logdir}")

checkpointing_training_loop(
    model,
    batch_size=batch_size,
    epochs=1000,
    manager=manager,
    epoch_var=epoch_var,
    step_var=step_var,
)

# %% [markdown]
# After the models have been saved, we can restore them using ```tf.train.Checkpoint.restore``` and assert that their performance corresponds to that logged during training.

# %%
for i, recorded_checkpoint in enumerate(manager.checkpoints):
    ckpt.restore(recorded_checkpoint)
    print(
        f"{i} restored model from epoch {int(epoch_var)} [step:{int(step_var)}] : ELBO training set {model.elbo(data)}"
    )

# %% [markdown]
# ## Copying (hyper)parameter values between models
#
# It is easy to interact with the set of all parameters of a model or a subcomponent programmatically.
#
# The following returns a dictionary of all parameters within

# %%
model = gpflow.models.SGPR(data, kernel=kernel, inducing_variable=inducing_variable)

# %%
gpflow.utilities.parameter_dict(model)

# %% [markdown]
# Such a dictionary can be assigned back to this model (or another model with the same tree of parameters) as follows:

# %%
params = gpflow.utilities.parameter_dict(model)
gpflow.utilities.multiple_assign(model, params)

# %% [markdown]
# ### TensorFlow `saved_model`
#
# At present, TensorFlow does not support saving custom variables like instances of the `gpflow.base.Parameter` class, see [this TensorFlow github issue](https://github.com/tensorflow/tensorflow/issues/34908).
#
# However, once training is complete, it is possible to clone the model and replace all `gpflow.base.Parameter`s with `tf.constant`s holding the same value:

# %%
model

# %%
frozen_model = gpflow.utilities.freeze(model)

# %% [markdown]
# In order to save the model we need to define a `tf.Module` holding the `tf.function`'s that we wish to export, as well as a reference to the underlying model:

# %%
module_to_save = tf.Module()
predict_fn = tf.function(
    frozen_model.predict_f,
    input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    autograph=False,
)
module_to_save.predict = predict_fn

# %% [markdown]
# Save original result for futher comparison

# %%
original_result = module_to_save.predict(samples_input)

# %% [markdown]
# Let's save the module
# %%
save_dir = str(pathlib.Path(tempfile.gettempdir()))
tf.saved_model.save(module_to_save, save_dir)

# %% [markdown]
# Load module back as new instance and compare predict results

# %%
loaded_model = tf.saved_model.load(save_dir)
loaded_result = loaded_model.predict(samples_input)

np.testing.assert_array_equal(loaded_result, original_result)
