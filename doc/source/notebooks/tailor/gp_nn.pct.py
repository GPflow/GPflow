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
# # Mixing TensorFlow models with GPflow
#
# This notebook explores the combination of Keras TensorFlow neural networks with GPflow models.

# %%
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import gpflow
from gpflow.ci_utils import ci_niter
from scipy.cluster.vq import kmeans2

from typing import Dict, Optional, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
import gpflow
from gpflow.utilities import to_default_float

iterations = ci_niter(100)

# %% [markdown]
# ## Convolutional network inside a GPflow model

# %%
original_dataset, info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True)
total_num_data = info.splits["train"].num_examples
image_shape = info.features["image"].shape
image_size = tf.reduce_prod(image_shape)
batch_size = 32


def map_fn(input_slice: Dict[str, tf.Tensor]):
    updated = input_slice
    image = to_default_float(updated["image"]) / 255.0
    label = to_default_float(updated["label"])
    return tf.reshape(image, [-1, image_size]), label


autotune = tf.data.experimental.AUTOTUNE
dataset = (
    original_dataset.shuffle(1024)
    .batch(batch_size, drop_remainder=True)
    .map(map_fn, num_parallel_calls=autotune)
    .prefetch(autotune)
    .repeat()
)


# %% [markdown]
# Here we'll use the GPflow functionality, but put a non-GPflow model inside the kernel.\
# Vanilla ConvNet. This gets 97.3% accuracy on MNIST when used on its own (+ final linear layer) after 20K iterations

# %%
class KernelWithConvNN(gpflow.kernels.Kernel):
    def __init__(
        self,
        image_shape: Tuple,
        output_dim: int,
        base_kernel: gpflow.kernels.Kernel,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        with self.name_scope:
            self.base_kernel = base_kernel
            input_size = int(tf.reduce_prod(image_shape))
            input_shape = (input_size,)

            self.cnn = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size),
                    tf.keras.layers.Reshape(image_shape),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=image_shape[:-1], padding="same", activation="relu"
                    ),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=(5, 5), padding="same", activation="relu"
                    ),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(output_dim, activation="relu"),
                    tf.keras.layers.Lambda(to_default_float),
                ]
            )

            self.cnn.build()

    def K(self, a_input: tf.Tensor, b_input: Optional[tf.Tensor] = None) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        transformed_b = self.cnn(b_input) if b_input is not None else b_input
        return self.base_kernel.K(transformed_a, transformed_b)

    def K_diag(self, a_input: tf.Tensor) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        return self.base_kernel.K_diag(transformed_a)


# %% [markdown]
# $K_{uf}$ is in ConvNN output space, therefore we need to update `Kuf` multidispatch.

# %%
class KernelSpaceInducingPoints(gpflow.inducing_variables.InducingPoints):
    pass


@gpflow.covariances.Kuu.register(KernelSpaceInducingPoints, KernelWithConvNN)
def Kuu(inducing_variable, kernel, jitter=None):
    func = gpflow.covariances.Kuu.dispatch(
        gpflow.inducing_variables.InducingPoints, gpflow.kernels.Kernel
    )
    return func(inducing_variable, kernel.base_kernel, jitter=jitter)


@gpflow.covariances.Kuf.register(KernelSpaceInducingPoints, KernelWithConvNN, object)
def Kuf(inducing_variable, kernel, a_input):
    return kernel.base_kernel(inducing_variable.Z, kernel.cnn(a_input))


# %% [markdown]
# Now we are ready to create and initialize the model:

# %%
num_mnist_classes = 10
output_dim = 5
num_inducing_points = 100
images_subset, labels_subset = next(iter(dataset.batch(32)))
images_subset = tf.reshape(images_subset, [-1, image_size])
labels_subset = tf.reshape(labels_subset, [-1, 1])

kernel = KernelWithConvNN(
    image_shape, output_dim, gpflow.kernels.SquaredExponential(), batch_size=batch_size
)

likelihood = gpflow.likelihoods.MultiClass(num_mnist_classes)

inducing_variable_kmeans = kmeans2(images_subset.numpy(), num_inducing_points, minit="points")[0]
inducing_variable_cnn = kernel.cnn(inducing_variable_kmeans)
inducing_variable = KernelSpaceInducingPoints(inducing_variable_cnn)

model = gpflow.models.SVGP(
    kernel,
    likelihood,
    inducing_variable=inducing_variable,
    num_data=total_num_data,
    num_latent_gps=num_mnist_classes,
)

# %% [markdown]
# And start optimization:

# %%
data_iterator = iter(dataset)
adam_opt = tf.optimizers.Adam(0.001)

training_loss = model.training_loss_closure(data_iterator)


@tf.function
def optimization_step():
    adam_opt.minimize(training_loss, var_list=model.trainable_variables)


for _ in range(iterations):
    optimization_step()

# %% [markdown]
# Let's do predictions after training. Don't expect that we will get a good accuracy, because we haven't run training for long enough.

# %%
m, v = model.predict_y(images_subset)
preds = np.argmax(m, 1).reshape(labels_subset.numpy().shape)
correct = preds == labels_subset.numpy().astype(int)
acc = np.average(correct.astype(float)) * 100.0

print("Accuracy is {:.4f}%".format(acc))
