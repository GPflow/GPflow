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

from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from ..base import Parameter
from ..utilities import positive
from ..utilities.ops import difference_matrix
from .base import Kernel
from .stationaries import IsotropicStationary, Stationary


class Periodic(Kernel):
    """
    The periodic family of kernels. Can be used to wrap any Stationary kernel
    to transform it into a periodic version. The canonical form (based on the
    SquaredExponential kernel) can be found in Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    The derivation can be achieved by mapping the original inputs through the
    transformation u = (cos(x), sin(x)).

    For the SquaredExponential base kernel, the result can be expressed as:

        k(r) =  σ² exp{ -0.5 sin²(π r / γ) / ℓ²}

    where:
    r is the Euclidean distance between the input points
    ℓ is the lengthscales parameter,
    σ² is the variance parameter,
    γ is the period parameter.

    NOTE: usually we have a factor of 4 instead of 0.5 in front but this
        is absorbed into the lengthscales hyperparameter.
    NOTE: periodic kernel uses `active_dims` of a base kernel, therefore
        the constructor doesn't have it as an argument.
    """

    def __init__(self, base_kernel: IsotropicStationary, period: Union[float, List[float]] = 1.0):
        """
        :param base_kernel: the base kernel to make periodic; must inherit from Stationary
            Note that `active_dims` should be specified in the base kernel.
        :param period: the period; to induce a different period per active dimension
            this must be initialized with an array the same length as the number
            of active dimensions e.g. [1., 1., 1.]
        """
        if not isinstance(base_kernel, IsotropicStationary):
            raise TypeError("Periodic requires an IsotropicStationary kernel as the `base_kernel`")

        super().__init__()
        self.base_kernel = base_kernel
        self.period = Parameter(period, transform=positive())
        self.base_kernel._validate_ard_active_dims(self.period)

    @property
    def active_dims(self):
        return self.base_kernel.active_dims

    @active_dims.setter
    def active_dims(self, value):
        self.base_kernel.active_dims = value

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        return self.base_kernel.K_diag(X)

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        r = np.pi * (difference_matrix(X, X2)) / self.period
        scaled_sine = tf.sin(r) / self.base_kernel.lengthscales
        if hasattr(self.base_kernel, "K_r"):
            sine_r = tf.reduce_sum(tf.abs(scaled_sine), -1)
            K = self.base_kernel.K_r(sine_r)
        else:
            sine_r2 = tf.reduce_sum(tf.square(scaled_sine), -1)
            K = self.base_kernel.K_r2(sine_r2)
        return K
