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

from ...base import MeanAndVariance
from ...experimental.check_shapes import check_shapes
from ...inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    MultioutputInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ...kernels import (
    IndependentLatent,
    LinearCoregionalization,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)
from ...posteriors import (
    FallbackIndependentLatentPosterior,
    FullyCorrelatedPosterior,
    IndependentPosteriorMultiOutput,
    LinearCoregionalizationPosterior,
)
from ..dispatch import conditional


@conditional._gpflow_internal_register(
    object, SharedIndependentInducingVariables, SharedIndependent, object
)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, broadcast L]",
    "f: [M, L]",
    "q_sqrt: [M_L_or_L_M_M...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def shared_independent_conditional(
    Xnew: tf.Tensor,
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    Multioutput conditional for an independent kernel and shared inducing inducing.
    Same behaviour as conditional with non-multioutput kernels.
    The covariance matrices used to calculate the conditional have the following shape:

    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: N or [N, N]

    Further reference:

    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.

    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, P]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
        Note: as we are using a independent kernel these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, P] or [P, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, P]
        - variance: [N, P], [P, N, N], [N, P, P] or [N, P, N, P]

        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    posterior = IndependentPosteriorMultiOutput(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
        precompute_cache=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


@conditional._gpflow_internal_register(
    object, SeparateIndependentInducingVariables, SeparateIndependent, object
)
@conditional._gpflow_internal_register(
    object, SharedIndependentInducingVariables, SeparateIndependent, object
)
@conditional._gpflow_internal_register(
    object, SeparateIndependentInducingVariables, SharedIndependent, object
)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, broadcast L]",
    "f: [M, L]",
    "q_sqrt: [M_L_or_L_M_M...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def separate_independent_conditional(
    Xnew: tf.Tensor,
    inducing_variable: MultioutputInducingVariables,
    kernel: MultioutputKernel,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    posterior = IndependentPosteriorMultiOutput(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
        precompute_cache=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


@conditional._gpflow_internal_register(
    object,
    (FallbackSharedIndependentInducingVariables, FallbackSeparateIndependentInducingVariables),
    IndependentLatent,
    object,
)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, broadcast L]",
    "f: [M, L]",
    "q_sqrt: [M_L_or_L_M_M...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def fallback_independent_latent_conditional(
    Xnew: tf.Tensor,
    inducing_variable: MultioutputInducingVariables,
    kernel: IndependentLatent,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """Interdomain conditional with independent latents.
    In this case the number of latent GPs (L) will be different than the number of outputs (P)
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [L, M, M]
    - Kuf: [M, L, N, P]
    - Kff: [N, P, N, P], [N, P, P], [N, P]

    Further reference:

    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    - See above for the parameters and the return value.
    """
    posterior = FallbackIndependentLatentPosterior(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
        precompute_cache=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


@conditional._gpflow_internal_register(object, InducingPoints, MultioutputKernel, object)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, broadcast L]",
    "f: [L, 1]",
    "q_sqrt: [L_1_or_1_L_L...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def inducing_point_conditional(
    Xnew: tf.Tensor,
    inducing_variable: InducingPoints,
    kernel: MultioutputKernel,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """Multi-output GP with fully correlated inducing variables.
    The inducing variables are shaped in the same way as evaluations of K, to allow a default
    inducing point scheme for multi-output kernels.
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, L, M, L]
    - Kuf: [M, L, N, P]
    - Kff: [N, P, N, P], [N, P, P], [N, P]

    Further reference:

    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.

    :param f: variational mean, [L, 1]
    :param q_sqrt: standard-deviations or cholesky, [L, 1]  or  [1, L, L]
    """
    posterior = FullyCorrelatedPosterior(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
        precompute_cache=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


@conditional._gpflow_internal_register(
    object,
    (SharedIndependentInducingVariables, SeparateIndependentInducingVariables),
    LinearCoregionalization,
    object,
)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, broadcast L]",
    "f: [M, L]",
    "q_sqrt: [M_L_or_L_M_M...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def coregionalization_conditional(
    Xnew: tf.Tensor,
    inducing_variable: MultioutputInducingVariables,
    kernel: LinearCoregionalization,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """Most efficient routine to project L independent latent gps through a mixing matrix W.
    The mixing matrix is a member of the `LinearCoregionalization` and has shape [P, L].
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [L, M, M]
    - Kuf: [L, M, N]
    - Kff: [L, N] or [L, N, N]

    Further reference:

    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    """
    posterior = LinearCoregionalizationPosterior(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
        precompute_cache=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
