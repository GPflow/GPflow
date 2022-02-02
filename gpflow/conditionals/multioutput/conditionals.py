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

from ...inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
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
def shared_independent_conditional(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """Multioutput conditional for an independent kernel and shared inducing inducing.
    Same behaviour as conditional with non-multioutput kernels.
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: N or [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    Parameters
    ----------
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
def separate_independent_conditional(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
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
def fallback_independent_latent_conditional(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """Interdomain conditional with independent latents.
    In this case the number of latent GPs (L) will be different than the number of outputs (P)
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [L, M, M]
    - Kuf: [M, L, N, P]
    - Kff: [N, P, N, P], [N, P, P], [N, P]

    Further reference
    -----------------
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
def inducing_point_conditional(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """Multi-output GP with fully correlated inducing variables.
    The inducing variables are shaped in the same way as evaluations of K, to allow a default
    inducing point scheme for multi-output kernels.
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, L, M, L]
    - Kuf: [M, L, N, P]
    - Kff: [N, P, N, P], [N, P, P], [N, P]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.

    Parameters
    ----------
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
def coregionalization_conditional(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """Most efficient routine to project L independent latent gps through a mixing matrix W.
    The mixing matrix is a member of the `LinearCoregionalization` and has shape [P, L].
    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [L, M, M]
    - Kuf: [L, M, N]
    - Kff: [L, N] or [L, N, N]

    Further reference
    -----------------
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
