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
from gpflow import kernels, test_utils

KERNEL_CLASSES = [
    kernel
    for cls in (kernels.Static, kernels.Stationary, kernels.Linear)
    for kernel in gpflow.ci_utils.subclasses(cls)
    if kernel not in (kernels.IsotropicStationary, kernels.AnisotropicStationary)
] + [kernels.ArcCosine]

rng = np.random.RandomState(42)


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
def test_kernel_interface(kernel_class: Type[kernels.Kernel]) -> None:
    """
    A valid kernel is positive semidefinite. Some kernels are only valid for
    particular input shapes, see https://github.com/GPflow/GPflow/issues/1328
    """
    N, N2, D = 101, 103, 5
    X = rng.randn(N, D)
    X2 = rng.randn(N2, D)
    kernel = kernel_class()

    if isinstance(kernel, kernels.White):
        # The White kernel is special in that it is based on indices, not
        # values, and hence White()(X, X2) is zero everywhere. This means we
        # need to explicitly check psd-ness of just kernel(X) itself.
        K = kernel(X)
        test_utils.assert_psd_matrix(K)
    else:
        test_utils.test_kernel(kernel, X, X2)


@pytest.mark.parametrize(
    "base_class", [kernel for kernel in gpflow.ci_utils.subclasses(kernels.IsotropicStationary)]
)
def test_positive_semidefinite_periodic(base_class: Type[kernels.IsotropicStationary]) -> None:
    """
    A valid kernel is positive semidefinite. Some kernels are only valid for
    particular input shapes, see https://github.com/GPflow/GPflow/issues/1328
    """
    kernel = kernels.Periodic(base_class())

    N, N2, D = 101, 103, 5
    X = rng.randn(N, D)
    X2 = rng.randn(N2, D)
    test_utils.test_kernel(kernel, X, X2)
