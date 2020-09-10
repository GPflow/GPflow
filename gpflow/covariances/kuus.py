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

import tensorflow as tf

from ..inducing_variables import InducingPoints, Multiscale, InducingPatches
from ..kernels import Kernel, SquaredExponential, Convolutional
from .dispatch import Kuu
from ..config import default_float


@Kuu.register(InducingPoints, Kernel)
def Kuu_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: Kernel, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(Multiscale, SquaredExponential)
def Kuu_sqexp_multiscale(inducing_variable: Multiscale, kernel: SquaredExponential, *, jitter=0.0):
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscales2 = tf.square(kernel.lengthscales + Zlen)
    sc = tf.sqrt(
        idlengthscales2[None, ...] + idlengthscales2[:, None, ...] - kernel.lengthscales ** 2
    )
    d = inducing_variable._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kernel.variance * tf.exp(-d / 2) * tf.reduce_prod(kernel.lengthscales / sc, 2)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(InducingPatches, Convolutional)
def Kuu_conv_patch(feat, kern, jitter=0.0):
    return kern.base_kernel.K(feat.Z) + jitter * tf.eye(len(feat), dtype=default_float())
