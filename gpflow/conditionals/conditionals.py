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

from ..config import default_jitter
from ..covariances import Kuf, Kuu
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..utilities.ops import eye
from .dispatch import conditional
from .util import base_conditional, expand_independent_outputs


@conditional.register(object, InducingVariables, Kernel, object)
def _conditional(
    Xnew: tf.Tensor,
    inducing_variable: InducingVariables,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """
    Single-output GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, R]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
           NOTE: as we are using a single-output kernel with repetitions
                 these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, R]
        - variance: [N, R], [R, N, N], [N, R, R] or [N, R, N, R]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    Kmm = Kuu(inducing_variable, kernel, jitter=default_jitter())  # [M, M]
    Kmn = Kuf(inducing_variable, kernel, Xnew)  # [M, N]
    Knn = kernel(Xnew, full_cov=full_cov)
    fmean, fvar = base_conditional(
        Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )  # [N, R],  [R, N, N] or [N, R]
    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)


@conditional.register(object, object, Kernel, object)
def _conditional(
    Xnew: tf.Tensor,
    X: tf.Tensor,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
):
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = 𝒩(𝟎, 𝐈)
        f = 𝐋v
    thus
        p(f) = 𝒩(𝟎, 𝐋𝐋ᵀ) = 𝒩(𝟎, 𝐊).
    In this case `f` represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).

    We assume R independent GPs, represented by the columns of f (and the
    first dimension of q_sqrt).

    :param Xnew: data matrix, size [N, D]. Evaluate the GP at these new points
    :param X: data points, size [M, D].
    :param kernel: GPflow kernel.
    :param f: data matrix, [M, R], representing the function values at X,
        for R functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, R] or [R, M, M].
    :param white: boolean of whether to use the whitened representation as
        described above.
    :return:
        - mean:     [N, R]
        - variance: [N, R] (full_cov = False), [R, N, N] (full_cov = True)
    """
    Kmm = kernel(X) + eye(tf.shape(X)[-2], value=default_jitter(), dtype=X.dtype)  # [..., M, M]
    Kmn = kernel(X, Xnew)  # [M, ..., N]
    Knn = kernel(Xnew, full_cov=full_cov)  # [..., N] (full_cov = False) or [..., N, N] (True)
    mean, var = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    return mean, var  # [N, R], [N, R] or [R, N, N]
