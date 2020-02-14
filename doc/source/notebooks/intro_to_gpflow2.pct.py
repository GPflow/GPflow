# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
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
from pathlib import Path

import datetime
import io
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.config import default_float

import warnings

warnings.filterwarnings('ignore')

# %% [markdown]
# Make `tensorboard` work inside notebook:

# %%
output_logdir = "/tmp/tensorboard"

# !rm -rf "{output_logdir}"
# !mkdir "{output_logdir}"

# %load_ext tensorboard
# %matplotlib inline


def enumerated_logdir(_logdir_id: int = [0]):
    logdir = Path(output_logdir, str(_logdir_id[0]))
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

plt.plot(X, Y, 'xk')
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
train_dataset = train_dataset.repeat()\
                    .prefetch(prefetch_size)\
                    .shuffle(buffer_size=shuffle_buffer_size)\
                    .batch(batch_size)

print(f"prefetch_size={prefetch_size}")
print(f"shuffle_buffer_size={shuffle_buffer_size}")
print(f"num_batches_per_epoch={num_batches_per_epoch}")

# %% [markdown]
# ## Define a GP model
#
# In GPflow 2.0, we use `tf.Module` (or the very thin `gpflow.base.Module` wrapper) to build all our models, as well as their components (kernels, likelihoods, parameters, and so on).

# %%
kernel = gpflow.kernels.SquaredExponential(variance=2.)
likelihood = gpflow.likelihoods.Gaussian()
inducing_variable = np.linspace(0, 10, num_features).reshape(-1, 1)

model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable)

# %% [markdown]
# You can set a module (or a particular parameter) to be non-trainable using the auxiliary method ```set_trainable(module, False)```:

# %%
from gpflow.utilities import set_trainable

set_trainable(likelihood, False)
set_trainable(kernel.variance, False)

set_trainable(likelihood, True)
set_trainable(kernel.variance, True)

# %% [markdown]
# We can use ```param.assign(value)``` to assign a value to a parameter:

# %%
kernel.lengthscale.assign(0.5)

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

with tf.GradientTape() as tape:
    tape.watch(model.trainable_variables)
    obj = - model.elbo(data)
    grads = tape.gradient(obj, model.trainable_variables)

optimizer.apply_gradients(zip(grads, model.trainable_variables))


# %% [markdown]
# For a more elaborate example of a gradient update we can define an ```optimization_step``` that uses the decorator ```tf.function``` on a closure. A closure is a callable that returns the model objective evaluated at a given dataset when called.

# %%
def optimization_step(model: gpflow.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor]):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        obj = - model.elbo(batch)
        grads = tape.gradient(obj, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# %% [markdown]
# We can use the functionality of TensorFlow Datasets to define a simple training loop that iterates over batches of the training dataset:

# %%
def simple_training_loop(model: gpflow.models.SVGP, epochs: int = 1, logging_epoch_freq: int = 10):
    batches = iter(train_dataset)
    tf_optimization_step = tf.function(optimization_step, autograph=False)
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
# We can monitor the training procedure using `tf.summary`. First we create a summary writer object through which we can write scalars and images.

# %%
from intro_to_gpflow2_plotting import plotting_regression, summary_matplotlib_image

samples_input = tf.cast(np.linspace(0, 10, 100).reshape(100, 1), default_float())

def monitored_training_loop(model: gpflow.models.SVGP, logdir: str,
                            epochs: int = 1, logging_epoch_freq: int = 10,
                            num_samples: int = 10):
    summary_writer = tf.summary.create_file_writer(logdir)
    tf_optimization_step = tf.function(optimization_step)
    batches = iter(train_dataset)

    with summary_writer.as_default():
        for epoch in range(epochs):
            for _ in range(num_batches_per_epoch):
                tf_optimization_step(model, next(batches))

            epoch_id = epoch + 1
            if epoch_id % logging_epoch_freq == 0:
                tf.print(f"Epoch {epoch_id}: ELBO (train) {model.elbo(data)}")

                mean, var = model.predict_f(samples_input)
                samples = model.predict_f_samples(samples_input, num_samples)
                fig = plotting_regression(X, Y, samples_input, mean, var, samples)

                summary_matplotlib_image(dict(model_samples=fig), step=epoch)
                tf.summary.scalar('elbo', data=model.elbo(data), step=epoch)
                tf.summary.scalar('likelihood/variance', data=model.likelihood.variance, step=epoch)
                tf.summary.scalar('kernel/lengthscale', data=model.kernel.lengthscale, step=epoch)
                tf.summary.scalar('kernel/variance', data=model.kernel.variance, step=epoch)


# %%
model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable)

output_logdir = enumerated_logdir()
monitored_training_loop(model, output_logdir, epochs=1000, logging_epoch_freq=100)

# %% [markdown]
# Then, we can use TensorBoard to examine the training procedure in more detail

# %%
# # %tensorboard --logdir "{output_logdir}"

# %% [markdown]
# ## Checkpointing: saving and loading models
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
model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable)

def checkpointing_training_loop(model: gpflow.models.SVGP,
                                batch_size: int,
                                epochs: int,
                                manager: tf.train.CheckpointManager,
                                logging_epoch_freq: int = 100,
                                epoch_var: Optional[tf.Variable] = None,
                                step_var: Optional[tf.Variable] = None):
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

checkpointing_training_loop(model, batch_size=batch_size, epochs=1000, manager=manager, epoch_var=epoch_var, step_var=step_var)

# %% [markdown]
# After the models have been saved, we can restore them using ```tf.train.Checkpoint.restore``` and assert that their performance corresponds to that logged during training.

# %%
for i, recorded_checkpoint in enumerate(manager.checkpoints):
    ckpt.restore(recorded_checkpoint)
    print(f"{i} restored model from epoch {int(epoch_var)} [step:{int(step_var)}] : ELBO training set {model.elbo(data)}")

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
