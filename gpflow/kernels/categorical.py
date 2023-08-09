# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from .. import default_int, set_trainable
from ..base import Parameter, TensorType
from . import Kernel


def latent_from_labels(Z: tf.Tensor, labels: TensorType) -> TensorType:
    """
    Converts a tensor of labels into a tensor of the latent space values of those labels.

    :param Z: Z is a tensor whose entries are the latent space values of the relevant label
        i.e. Z[3] is the value of label 3.
        [num_labels, label_dim=1]
    :param labels: labels is a vector.  Each entry corresponds to one of the points in the data,
        X.  The value is an integer that specifies which category that point belongs to.  Note
        that the value will likely be stored as a float (or whatever the rest of the data is)
        even though it represents an integer value.
    :return: [..., N, D + label_dim]
    """
    indices = tf.cast(labels, default_int())  # [..., N]
    return tf.gather(Z, indices)  # [..., N, label_dim]


def _concat_inputs_with_latents(Z: tf.Tensor, X: TensorType) -> TensorType:
    """
    Transforms a dataset X from category space into the latent space.

    :param Z: Z is a tensor whose entries are the latent space values of the relevant label
        i.e. Z[3] is the value of label 3.
        [num_labels]
    :param X: X is the input data.  The last column should contain the labels as integers.
        [..., N, D+1]
    :return: [..., N, D + label_dim]
    """
    labels = X[..., -1]
    latent_values = latent_from_labels(Z, labels)

    return tf.concat([X[..., :-1], latent_values], axis=-1)  # [..., N, x_dim + label_dim]


class Categorical(Kernel):
    """
    A class that implements a categorical latent space wrapper for other Kernels.  It works
     by dynamically replacing integer labels in the input data with values in a latent space (Z),
     wrapping two kernels which are combined multiplicatively and otherwise work as normal.
    Kernel 1 is any kernel of your choice which will deal with the non-categorical dimensions.
    Kernel 2 deals with the categorical dimension.  It is fixed in training, in order to reduce the number
     of degrees of freedom in optimisation.  You may also find fixing the lengthscale to be useful here.
     Note that Z is parameterised by the differences of latent space values for each category for the same reason.
    Multiple categories can be included by wrapping multiple layers of CategoricalKernel.

    :param non_categorical_kernel: The non-categorical kernel.
    :param categorical_kernel: The categorical kernel.  Set its active_dims to the label dimension.
    :param num_labels: The number of labels
    """

    def __init__(
        self,
        non_categorical_kernel: Kernel,
        categorical_kernel: Kernel,
        num_labels: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        set_trainable(categorical_kernel, False)
        self.wrapped_kernel = non_categorical_kernel * categorical_kernel
        label_dim = 1
        # parametrise by the `num_labels - 1` differences of latent space values
        self._Z_deltas = Parameter(
            np.random.random((num_labels - 1, label_dim)) * categorical_kernel.lengthscales * 10
        )
        super().__init__(*args, **kwargs)

    def _concat_inputs_with_latents(self, X: TensorType) -> TensorType:
        return _concat_inputs_with_latents(self.Z, X)

    @property
    def Z(self) -> tf.Tensor:
        """
        Z is a tensor whose entries are the latent space values of the relevant label
        i.e. Z[3] is the value of label 3.
        """
        # This sets Z[0] = 0 and then creates Z[1:] by adding the deltas.
        #  This is achieved by using a lower diagonal matrix of ones,
        #  so Z[1] = _Z_deltas[0], Z[2] = _Z_deltas[0] + _Z_deltas[1], etc.
        Z = tf.concat([tf.constant(0, shape=(1,), dtype=tf.float64), tf.squeeze(self._Z_deltas)], 0)
        m = tf.linalg.band_part(tf.ones([tf.size(Z), tf.size(Z)], dtype=tf.float64), -1, 0)
        return tf.expand_dims(tf.linalg.matvec(m, Z), -1)

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return self.wrapped_kernel.K(
            self._concat_inputs_with_latents(X),
            self._concat_inputs_with_latents(X2) if X2 is not None else None,
        )

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self.wrapped_kernel.K_diag(self._concat_inputs_with_latents(X))
