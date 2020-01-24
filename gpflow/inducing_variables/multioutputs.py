# Copyright 2018 GPflow authors
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

from typing import Union

import tensorflow as tf

from ..kernels import (MultioutputKernel, SeparateIndependent, LinearCoregionalization, SharedIndependent,
                       IndependentLatent)
from ..covariances import Kuf, Kuu
from . import InducingVariables, InducingPoints


class MultioutputInducingVariables(InducingVariables):
    """
    Multioutput Inducing Variables
    Base class for methods which define a collection of inducing variables which
    in some way can be grouped. The main example is where the inducing variables
    consist of outputs of various independent GPs. This can be because our model
    uses multiple independent GPs (SharedIndependent, SeparateIndependent) or
    because it is constructed from independent GPs (eg IndependentLatent,
    LinearCoregionalization).
    """
    pass


class FallbackSharedIndependentInducingVariables(MultioutputInducingVariables):
    """
    Shared definition of inducing variables for each independent latent process.

    This class is designated to be used to
     - provide a general interface for multioutput kernels
       constructed from independent latent processes,
     - only require the specification of Kuu and Kuf.
    All multioutput kernels constructed from independent latent processes allow
    the inducing variables to be specified in the latent processes, and a
    reasonably efficient method (i.e. one that takes advantage of the
    independence in the latent processes) can be specified quite generally by
    only requiring the following covariances:
     - Kuu: [L, M, M],
     - Kuf: [L, M, N, P].
    In `mo_conditionals.py` we define a conditional() implementation for this
    combination. We specify this code path for all kernels which inherit from
    `IndependentLatentBase`. This set-up allows inference with any such kernel
    to be implemented by specifying only `Kuu()` and `Kuf()`.

    We call this the base class, since many multioutput GPs that are constructed
    from independent latent processes acutally allow even more efficient
    approximations. However, we include this code path, as it does not require
    specifying a new `conditional()` implementation.

    Here, we share the definition of inducing variables between all latent
    processes.
    """

    def __init__(self, inducing_variable):
        super().__init__()
        self.inducing_variable_shared = inducing_variable

    def __len__(self):
        return len(self.inducing_variable_shared)


class FallbackSeparateIndependentInducingVariables(MultioutputInducingVariables):
    """
    Separate set of inducing variables for each independent latent process.

    This class is designated to be used to
     - provide a general interface for multioutput kernels
       constructed from independent latent processes,
     - only require the specification of Kuu and Kuf.
    All multioutput kernels constructed from independent latent processes allow
    the inducing variables to be specified in the latent processes, and a
    reasonably efficient method (i.e. one that takes advantage of the
    independence in the latent processes) can be specified quite generally by
    only requiring the following covariances:
     - Kuu: [L, M, M],
     - Kuf: [L, M, N, P].
    In `mo_conditionals.py` we define a conditional() implementation for this
    combination. We specify this code path for all kernels which inherit from
    `IndependentLatentBase`. This set-up allows inference with any such kernel
    to be implemented by specifying only `Kuu()` and `Kuf()`.

    We call this the base class, since many multioutput GPs that are constructed
    from independent latent processes acutally allow even more efficient
    approximations. However, we include this code path, as it does not require
    specifying a new `conditional()` implementation.

    We use a different definition of inducing variables for each latent process.
    Note: each object should have the same number of inducing variables, M.
    """

    def __init__(self, inducing_variable_list):
        super().__init__()
        self.inducing_variable_list = inducing_variable_list

    def __len__(self):
        return len(self.inducing_variable_list[0])


class SharedIndependentInducingVariables(FallbackSharedIndependentInducingVariables):
    """
    Here, we define the same inducing variables as in the base class. However,
    this class is intended to be used without the constraints on the shapes that
    `Kuu()` and `Kuf()` return. This allows a custom `conditional()` to provide
    the most efficient implementation.
    """
    pass


class SeparateIndependentInducingVariables(FallbackSeparateIndependentInducingVariables):
    """
    Here, we define the same inducing variables as in the base class. However,
    this class is intended to be used without the constraints on the shapes that
    `Kuu()` and `Kuf()` return. This allows a custom `conditional()` to provide
    the most efficient implementation.
    """
    pass


@Kuf.register(InducingPoints, MultioutputKernel, object)
def _Kuf(inducing_variable: InducingPoints, kernel: MultioutputKernel, Xnew: tf.Tensor):
    return kernel(inducing_variable.Z, Xnew, full=True, full_output_cov=True)  # [M, P, N, P]


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def _Kuf(inducing_variable: SharedIndependentInducingVariables, kernel: SharedIndependent, Xnew: tf.Tensor):
    return Kuf(inducing_variable.inducing_variable_shared, kernel.kernel, Xnew)  # [M, N]


@Kuf.register(SeparateIndependentInducingVariables, SharedIndependent, object)
def _Kuf(inducing_variable: SeparateIndependentInducingVariables, kernel: SharedIndependent, Xnew: tf.Tensor):
    return tf.stack([Kuf(f, kernel.kernel, Xnew)
                     for f in inducing_variable.inducing_variable_list], axis=0)  # [L, M, N]


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(inducing_variable: SharedIndependentInducingVariables, kernel: SeparateIndependent, Xnew: tf.Tensor):
    return tf.stack([Kuf(inducing_variable.inducing_variable_shared, k, Xnew) for k in kernel.kernels], axis=0)  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(inducing_variable: SeparateIndependentInducingVariables, kernel: SeparateIndependent, Xnew: tf.Tensor):
    Kufs = [Kuf(f, k, Xnew) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)]
    return tf.stack(Kufs, axis=0)  # [L, M, N]


@Kuf.register((FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables),
              LinearCoregionalization,
              object)
def _Kuf(inducing_variable: Union[SeparateIndependentInducingVariables, SharedIndependentInducingVariables],
         kernel: LinearCoregionalization, Xnew: tf.Tensor):
    kuf_impl = Kuf.dispatch(type(inducing_variable), SeparateIndependent, object)
    K = tf.transpose(kuf_impl(inducing_variable, kernel, Xnew), [1, 0, 2])  # [M, L, N]
    return K[:, :, :, None] * tf.transpose(kernel.W)[None, :, None, :]  # [M, L, N, P]


@Kuf.register(SharedIndependentInducingVariables, LinearCoregionalization, object)
def _Kuf(inducing_variable: SharedIndependentInducingVariables, kernel: SeparateIndependent, Xnew: tf.Tensor):
    return tf.stack([Kuf(inducing_variable.inducing_variable_shared, k, Xnew) for k in kernel.kernels], axis=0)  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, LinearCoregionalization, object)
def _Kuf(inducing_variable, kernel, Xnew):
    return tf.stack([Kuf(f, k, Xnew)
                     for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)], axis=0)  # [L, M, N]


@Kuu.register(InducingPoints, MultioutputKernel)
def _Kuu(inducing_variable: InducingPoints, kernel: MultioutputKernel, *, jitter=0.0):
    Kmm = kernel(inducing_variable.Z, full=True, full_output_cov=True)  # [M, P, M, P]
    M = tf.shape(Kmm)[0] * tf.shape(Kmm)[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=Kmm.dtype), tf.shape(Kmm))
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
def _Kuu(inducing_variable: FallbackSharedIndependentInducingVariables, kernel: SharedIndependent, *, jitter=0.0):
    Kmm = Kuu(inducing_variable.inducing_variable_shared, kernel.kernel)  # [M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype) * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, (SeparateIndependent, IndependentLatent))
def _Kuu(inducing_variable: FallbackSharedIndependentInducingVariables,
         kernel: Union[SeparateIndependent, IndependentLatent],
         *,
         jitter=0.0):
    Kmm = tf.stack([Kuu(inducing_variable.inducing_variable_shared, k) for k in kernel.kernels], axis=0)  # [L, M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSeparateIndependentInducingVariables, SharedIndependent)
def _Kuu(inducing_variable: FallbackSeparateIndependentInducingVariables, kernel: SharedIndependent, *, jitter=0.0):
    Kmm = tf.stack([Kuu(f, kernel.kernel) for f in inducing_variable.inducing_variable_list], axis=0)  # [L, M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSeparateIndependentInducingVariables, (SeparateIndependent, LinearCoregionalization))
def _Kuu(inducing_variable: FallbackSeparateIndependentInducingVariables,
         kernel: Union[SeparateIndependent, LinearCoregionalization],
         *,
         jitter=0.0):
    Kmms = [Kuu(f, k) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)]
    Kmm = tf.stack(Kmms, axis=0)  # [L, M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat
