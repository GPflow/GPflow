from typing import Union

import tensorflow as tf

from ..inducing_variables import (InducingPoints, FallbackSharedIndependentInducingVariables,
                                  FallbackSeparateIndependentInducingVariables, SharedIndependentInducingVariables,
                                  SeparateIndependentInducingVariables)
from ..kernels import (MultioutputKernel, SeparateIndependent, LinearCoregionalisation, SharedIndependent)
from .dispatch import Kuf


@Kuf.register(InducingPoints, MultioutputKernel, object)
def _Kuf(feature: InducingPoints, kernel: MultioutputKernel, Xnew: tf.Tensor):
    return kernel(feature.Z, Xnew, full=True, full_output_cov=True)  # [M, P, N, P]


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def _Kuf(feature: SharedIndependentInducingVariables, kernel: SharedIndependent, Xnew: tf.Tensor):
    return Kuf(feature.inducing_variable, kernel.kernel, Xnew)  # [M, N]


@Kuf.register(SeparateIndependentInducingVariables, SharedIndependent, object)
def _Kuf(feature: SeparateIndependentInducingVariables, kernel: SharedIndependent, Xnew: tf.Tensor):
    return tf.stack([Kuf(f, kernel.kernel, Xnew) for f in feature.inducing_variable_list], axis=0)  # [L, M, N]


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(feature: SharedIndependentInducingVariables, kernel: SeparateIndependent, Xnew: tf.Tensor):
    return tf.stack([Kuf(feature.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0)  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(feature: SeparateIndependentInducingVariables, kernel: SeparateIndependent, Xnew: tf.Tensor):
    Kufs = [Kuf(f, k, Xnew) for f, k in zip(feature.inducing_variable_list, kernel.kernels)]
    return tf.stack(Kufs, axis=0)  # [L, M, N]


@Kuf.register((FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables), LinearCoregionalisation,
              object)
def _Kuf(feature: Union[SeparateIndependentInducingVariables, SharedIndependentInducingVariables],
         kernel: LinearCoregionalisation, Xnew: tf.Tensor):
    kuf_impl = Kuf.dispatch(type(feature), SeparateIndependent, object)
    K = tf.transpose(kuf_impl(feature, kernel, Xnew), [1, 0, 2])  # [M, L, N]
    return K[:, :, :, None] * tf.transpose(kernel.W)[None, :, None, :]  # [M, L, N, P]


@Kuf.register(SharedIndependentInducingVariables, LinearCoregionalisation, object)
def _Kuf(feature: SharedIndependentInducingVariables, kernel: SeparateIndependent, Xnew: tf.Tensor):
    return tf.stack([Kuf(feature.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0)  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, LinearCoregionalisation, object)
def _Kuf(feature, kernel, Xnew):
    return tf.stack([Kuf(f, k, Xnew) for f, k in zip(feature.features, kernel.kernels)], axis=0)  # [
    # L, M, N]
