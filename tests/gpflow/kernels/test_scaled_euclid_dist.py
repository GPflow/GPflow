# Copyright 2018 the GPflow authors.
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

import numpy as np
import pytest
import tensorflow as tf

import gpflow.kernels as kernels
from gpflow.base import AnyNDArray

rng = np.random.RandomState(0)


class Datum:
    num_data = 100
    D = 100
    X: AnyNDArray = rng.rand(num_data, D) * 100


kernel_list = [
    kernels.Matern12(),
    kernels.Matern32(),
    kernels.Matern52(),
    kernels.Exponential(),
    kernels.Cosine(),
]


@pytest.mark.parametrize("kernel", kernel_list)
def test_kernel_euclidean_distance(kernel: kernels.Kernel) -> None:
    """
    Tests output & gradients of kernels that are a function of the (scaled) euclidean distance
    of the points. We test on a high dimensional space, which can generate very small distances
    causing the scaled_square_dist to generate some negative values.
    """
    K = kernel(Datum.X)
    assert not np.isnan(K).any(), "NaNs in the output of the " + kernel.__name__ + "kernel."
    assert np.isfinite(K).all(), "Infs in the output of the " + kernel.__name__ + " kernel."

    X_as_param = tf.Variable(Datum.X)
    with tf.GradientTape() as tape:
        K_value = kernel(X_as_param, X_as_param)
    dK = tape.gradient(K_value, X_as_param)[0]

    assert not np.isnan(dK).any(), "NaNs in the gradient of the " + kernel.__name__ + " kernel."
    assert np.isfinite(dK).all(), "Infs in the output of the " + kernel.__name__ + " kernel."
