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

from ...base import SamplesMeanAndVariance
from ...experimental.check_shapes import check_shapes
from ...inducing_variables import (
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ...kernels import LinearCoregionalization, SeparateIndependent
from ..dispatch import conditional, sample_conditional
from ..util import mix_latent_gp, sample_mvn


@sample_conditional.register(
    object, SharedIndependentInducingVariables, LinearCoregionalization, object
)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, maybe_R...]",
    "f: [M, R]",
    "return[0]: [batch..., N, P] if num_samples is None",
    "return[0]: [batch..., num_samples, N, P] if num_samples is not None",
    "return[1]: [batch..., N, P]",
    "return[2]: [batch..., N, P]",
)
def _sample_conditional(
    Xnew: tf.Tensor,
    inducing_variable: SharedIndependentInducingVariables,
    kernel: LinearCoregionalization,
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
     However, for some combinations of Mok and Mof, more efficient sampling routines exist.
     The dispatcher will make sure that we use the most efficent one.

    :return: samples, mean, cov
    """
    if full_cov:
        raise NotImplementedError("full_cov not yet implemented")
    if full_output_cov:
        raise NotImplementedError("full_output_cov not yet implemented")

    ind_conditional = conditional.dispatch_or_raise(
        object, SeparateIndependentInducingVariables, SeparateIndependent, object
    )
    g_mu, g_var = ind_conditional(
        Xnew, inducing_variable, kernel, f, white=white, q_sqrt=q_sqrt
    )  # [..., N, L], [..., N, L]
    g_sample = sample_mvn(g_mu, g_var, full_cov, num_samples=num_samples)  # [..., (S), N, L]
    f_mu, f_var = mix_latent_gp(kernel.W, g_mu, g_var, full_cov, full_output_cov)
    f_sample = tf.tensordot(g_sample, kernel.W, [[-1], [-1]])  # [..., N, P]
    return f_sample, f_mu, f_var
