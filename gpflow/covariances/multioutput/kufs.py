from typing import Union

import tensorflow as tf

from ...inducing_variables import (
    InducingPoints,
    FallbackSharedIndependentInducingVariables,
    FallbackSeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
    SeparateIndependentInducingVariables,
)
from ...kernels import (
    MultioutputKernel,
    SeparateIndependent,
    LinearCoregionalization,
    SharedIndependent,
)
from ..dispatch import Kuf


@Kuf.register(InducingPoints, MultioutputKernel, object)
def _Kuf(inducing_variable: InducingPoints, kernel: MultioutputKernel, Xnew: tf.Tensor):
    return kernel(inducing_variable.Z, Xnew, full_cov=True, full_output_cov=True)  # [M, P, N, P]


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def _Kuf(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
):
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]


@Kuf.register(SeparateIndependentInducingVariables, SharedIndependent, object)
def _Kuf(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
):
    return tf.stack(
        [Kuf(f, kernel.kernel, Xnew) for f in inducing_variable.inducing_variable_list], axis=0
    )  # [L, M, N]


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: tf.Tensor,
):
    return tf.stack(
        [Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0
    )  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: tf.Tensor,
):
    Kufs = [
        Kuf(f, k, Xnew) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)
    ]
    return tf.stack(Kufs, axis=0)  # [L, M, N]


@Kuf.register(
    (FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables),
    LinearCoregionalization,
    object,
)
def _Kuf(
    inducing_variable: Union[
        SeparateIndependentInducingVariables, SharedIndependentInducingVariables
    ],
    kernel: LinearCoregionalization,
    Xnew: tf.Tensor,
):
    kuf_impl = Kuf.dispatch(type(inducing_variable), SeparateIndependent, object)
    K = tf.transpose(kuf_impl(inducing_variable, kernel, Xnew), [1, 0, 2])  # [M, L, N]
    return K[:, :, :, None] * tf.transpose(kernel.W)[None, :, None, :]  # [M, L, N, P]


@Kuf.register(SharedIndependentInducingVariables, LinearCoregionalization, object)
def _Kuf(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: tf.Tensor,
):
    return tf.stack(
        [Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0
    )  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, LinearCoregionalization, object)
def _Kuf(inducing_variable, kernel, Xnew):
    return tf.stack(
        [Kuf(f, k, Xnew) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)],
        axis=0,
    )  # [L, M, N]
