# noqa: F811

import tensorflow as tf

from ..covariances import Kuf, Kuu
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..utilities.ops import eye
from ..config import default_jitter
from .dispatch import conditional
from .util import base_conditional, expand_independent_outputs


@conditional.register(object, InducingVariables, Kernel, object)
def _conditional(Xnew: tf.Tensor,
                 inducing_variable: InducingVariables,
                 kernel: Kernel,
                 function: tf.Tensor,
                 *,
                 full_cov=False,
                 full_output_cov=False,
                 q_sqrt=None,
                 white=False):
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
    Knn = kernel(Xnew, full=full_cov)
    fmean, fvar = base_conditional(Kmn,
                                   Kmm,
                                   Knn,
                                   function,
                                   full_cov=full_cov,
                                   q_sqrt=q_sqrt,
                                   white=white)  # [N, R],  [R, N, N] or [N, R]
    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)


@conditional.register(object, object, Kernel, object)
def _conditional(Xnew: tf.Tensor,
                 X: tf.Tensor,
                 kernel: Kernel,
                 function: tf.Tensor,
                 *,
                 full_cov=False,
                 full_output_cov=False,
                 q_sqrt=None,
                 white=False):
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
    Kmm = kernel(X) + eye(tf.shape(X)[-2], value=default_jitter(), dtype=X.dtype)
    Kmn = kernel(X, Xnew)
    Knn = kernel(Xnew, full=full_cov)
    mean, var = base_conditional(Kmn,
                                 Kmm,
                                 Knn,
                                 function,
                                 full_cov=full_cov,
                                 q_sqrt=q_sqrt,
                                 white=white)

    return mean, var  # [N, R], [N, R] or [R, N, N]
