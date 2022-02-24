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

from typing import Type

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_array_less

import gpflow.ci_utils
from gpflow import kernels

KERNEL_CLASSES = [
    kernel
    for cls in (kernels.Static, kernels.Stationary, kernels.Linear)
    for kernel in gpflow.ci_utils.subclasses(cls)
    if kernel not in (kernels.IsotropicStationary, kernels.AnisotropicStationary)
] + [kernels.ArcCosine]

rng = np.random.RandomState(42)


def pos_semidefinite(kernel: kernels.Kernel) -> None:
    N, D = 100, 5
    X = rng.randn(N, D)

    cov = kernel(X)
    eig = tf.linalg.eigvalsh(cov).numpy()
    assert_array_less(-1e-12, eig)


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
def test_positive_semidefinite(kernel_class: Type[kernels.Kernel]) -> None:
    """
    A valid kernel is positive semidefinite. Some kernels are only valid for
    particular input shapes, see https://github.com/GPflow/GPflow/issues/1328
    """
    kernel = kernel_class()
    pos_semidefinite(kernel)


@pytest.mark.parametrize(
    "base_class", [kernel for kernel in gpflow.ci_utils.subclasses(kernels.IsotropicStationary)]
)
def test_positive_semidefinite_periodic(base_class: Type[kernels.IsotropicStationary]) -> None:
    """
    A valid kernel is positive semidefinite. Some kernels are only valid for
    particular input shapes, see https://github.com/GPflow/GPflow/issues/1328
    """
    kernel = kernels.Periodic(base_class())
    pos_semidefinite(kernel)
