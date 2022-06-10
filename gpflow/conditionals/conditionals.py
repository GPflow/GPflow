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

from ..base import MeanAndVariance
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..posteriors import BasePosterior, VGPPosterior, get_posterior_class
from .dispatch import conditional


@conditional._gpflow_internal_register(object, InducingVariables, Kernel, object)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "inducing_variable: [M, D, broadcast R]",
    "f: [M, R]",
    "q_sqrt: [M_R_or_R_M_M...]",
    "return[0]: [batch..., N, R]",
    "return[1]: [batch..., N, R] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., R, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, R, R] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, R, N, R] if full_cov and full_output_cov",
)
def _sparse_conditional(
    Xnew: tf.Tensor,
    inducing_variable: InducingVariables,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    Single-output GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:

    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: [N, N]

    Further reference:

    - See `gpflow.conditionals._dense_conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    :param Xnew: data matrix
    :param f: data matrix
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
        NOTE: as we are using a single-output kernel with repetitions
        these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return: mean and variance
    """
    posterior_class = get_posterior_class(kernel, inducing_variable)

    posterior: BasePosterior = posterior_class(
        kernel,
        inducing_variable,
        f,
        q_sqrt,
        whiten=white,
        mean_function=None,
        precompute_cache=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


@conditional._gpflow_internal_register(object, object, Kernel, object)
@check_shapes(
    "Xnew: [batch..., N, D]",
    "X: [M, D]",
    "f: [M, R]",
    "return[0]: [batch..., N, R]",
    "return[1]: [batch..., N, R] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., R, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, R, R] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, R, N, R] if full_cov and full_output_cov",
)
def _dense_conditional(
    Xnew: tf.Tensor,
    X: tf.Tensor,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that::

        p(v) = 𝒩(𝟎, 𝐈)
        f = 𝐋v

    thus::

        p(f) = 𝒩(𝟎, 𝐋𝐋ᵀ) = 𝒩(𝟎, 𝐊).

    In this case `f` represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).

    We assume R independent GPs, represented by the columns of f (and the
    first dimension of q_sqrt).

    :param Xnew: data matrix. Evaluate the GP at these new points
    :param X: data points.
    :param kernel: GPflow kernel.
    :param f: data matrix, representing the function values at X,
        for R functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation as
        described above.
    :return: mean and variance
    """
    posterior = VGPPosterior(
        kernel=kernel,
        X=X,
        q_mu=f,
        q_sqrt=q_sqrt,
        white=white,
        precompute_cache=None,
    )
    return posterior.fused_predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
