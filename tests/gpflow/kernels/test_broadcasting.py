# Copyright 2020 the GPflow authors.
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

import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose
import pytest

import gpflow
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
    # Following kernels do not broadcast:
    pytest.param(kernels.ArcCosine, marks=pytest.mark.xfail),  # broadcasting not implemented
    pytest.param(kernels.Coregion, marks=pytest.mark.xfail),  # broadcasting not implemented
    pytest.param(kernels.Periodic, marks=pytest.mark.xfail),  # broadcasting not implemented
]


def _test_no_active_dims(Kernel):
    S, N, M, D = 5, 4, 3, 2
    X1 = np.random.randn(S, N, D)
    X2 = np.random.randn(M, D)
    kernel = Kernel() + gpflow.kernels.White()

    compare_vs_map(X1, X2, kernel)


def _test_slice_active_dims(Kernel):
    S, N, M, D = 5, 4, 3, 4
    d = 2
    X1 = np.random.randn(S, N, D)
    X2 = np.random.randn(M, D)
    kernel = Kernel(active_dims=slice(1, 1 + d))

    compare_vs_map(X1, X2, kernel)


def _test_indices_active_dims(Kernel):
    S, N, M, D = 5, 4, 3, 4

    X1 = np.random.randn(S, N, D)
    X2 = np.random.randn(M, D)
    kernel = Kernel(active_dims=[1, 3])

    compare_vs_map(X1, X2, kernel)


def compare_vs_map(X1, X2, kernel):
    K12_loop = tf.stack([kernel(x, X2) for x in X1])  # [S, N, M]
    K12_native = kernel(X1, X2)  # [S, N, M]
    assert_allclose(K12_loop.numpy(), K12_native.numpy())

    K11_loop = tf.stack([kernel(x) for x in X1])
    K11_native = kernel(X1)
    assert_allclose(K11_loop.numpy(), K11_native.numpy())

    K1_loop = tf.stack([kernel(x, full=False) for x in X1])
    K1_native = kernel(X1, full=False)
    assert_allclose(K1_loop.numpy(), K1_native.numpy())


def test_squaredexponential():
    _test_no_active_dims(gpflow.kernels.SquaredExponential)


def test_squaredexponential_slice():
    _test_slice_active_dims(gpflow.kernels.SquaredExponential)


def test_squaredexponential_indices():
    _test_indices_active_dims(gpflow.kernels.SquaredExponential)


@pytest.mark.parametrize("Kernel", KERNEL_CLASSES)
def test_all_no_active_dims(Kernel):
    _test_no_active_dims(Kernel)
