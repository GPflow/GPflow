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

from typing import Optional

import tensorflow as tf

from ..base import SamplesMeanAndVariance
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from .dispatch import conditional, sample_conditional
from .util import sample_mvn


@sample_conditional.register(object, object, Kernel, object)
@sample_conditional.register(object, InducingVariables, Kernel, object)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, maybe_R...]",
    "f: [M, R]",
    "return[0]: [batch..., N, R] if num_samples is None",
    "return[0]: [batch..., num_samples, N, R] if num_samples is not None",
    "return[1]: [batch..., N, R]",
    "return[2]: [batch..., N, R] if (not full_cov) and (not full_output_cov)",
    "return[2]: [batch..., R, N, N] if full_cov and (not full_output_cov)",
    "return[2]: [batch..., N, R, R] if (not full_cov) and full_output_cov",
)
def _sample_conditional(
    Xnew: tf.Tensor,
    inducing_variable: InducingVariables,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
    num_samples: Optional[int] = None,
) -> SamplesMeanAndVariance:
    """
    `sample_conditional` will return a sample from the conditional distribution.
    In most cases this means calculating the conditional mean m and variance v and then
    returning m + sqrt(v) * eps, with eps ~ N(0, 1).
    However, for some combinations of Mok and Mof more efficient sampling routines exists.
    The dispatcher will make sure that we use the most efficient one.

    :return: samples, mean, cov
    """

    if full_cov and full_output_cov:
        msg = "The combination of both `full_cov` and `full_output_cov` is not permitted."
        raise NotImplementedError(msg)

    mean, cov = conditional(
        Xnew,
        inducing_variable,
        kernel,
        f,
        q_sqrt=q_sqrt,
        white=white,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
    )
    if full_cov:
        # mean: [..., N, P]
        # cov: [..., P, N, N]
        mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
        samples = sample_mvn(
            mean_for_sample, cov, full_cov=True, num_samples=num_samples
        )  # [..., (S), P, N]
        samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
    else:
        # mean: [..., N, P]
        # cov: [..., N, P] or [..., N, P, P]
        samples = sample_mvn(
            mean, cov, full_cov=full_output_cov, num_samples=num_samples
        )  # [..., (S), N, P]

    return samples, mean, cov
