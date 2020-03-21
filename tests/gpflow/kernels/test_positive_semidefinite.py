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
from numpy.testing import assert_array_less

from gpflow import kernels

KERNEL_CLASSES = [
    # Static
    kernels.White,
    kernels.Constant,
    # Stationary
    kernels.SquaredExponential,
    kernels.RationalQuadratic,
    kernels.Exponential,
    kernels.Matern12,
    kernels.Matern32,
    kernels.Matern52,
    kernels.Cosine,
    kernels.Linear,
    kernels.Polynomial,
]

rng = np.random.RandomState(42)


@pytest.mark.parametrize("kernel", KERNEL_CLASSES)
def test_positive_semidefinite(kernel):
    N, D = 100, 5
    X = rng.randn(N, D)
    kern = kernel()

    cov = kern(X)
    eig = tf.linalg.eigvalsh(cov).numpy()
    assert_array_less(-np.finfo(eig.dtype).eps * 1e4, eig)
