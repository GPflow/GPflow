# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews
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

from . import features, mean_functions, settings, misc
from .decors import name_scope
from .dispatch import conditional, sample_conditional
from .expectations import expectation
from .features import InducingFeature, InducingPoints, Kuf, Kuu
from .kernels import Kernel
from .probability_distributions import Gaussian

logger = settings.logger()


# -----------
# CONDITIONAL
# -----------

@conditional.register(object, InducingFeature, Kernel, object)
@name_scope("conditional")
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    Single-output GP conditional.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: M x M
    - Kuf: M x N
    - Kff: N or N x N

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` (below) for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size N x D.
    :param feat: gpflow.InducingFeature object
    :param kern: gpflow kernel object.
    :param f: data matrix, M x R
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
     Note: as we are using a single-output kernel with repetitions these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     N x R
        - variance: N x R, R x N x N, N x R x R or N x R x N x R
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    logger.debug("Conditional: Inducing Feature - Kernel")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x N
    Knn = kern.K(Xnew) if full_cov else kern.Kdiag(Xnew)

    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                   q_sqrt=q_sqrt, white=white)  # N x R,  R x N x N or N x R
    return fmean, _expand_independent_outputs(fvar, full_cov, full_output_cov)


@conditional.register(object, object, Kernel, object)
@name_scope("conditional")
def _conditional(Xnew, X, kern, f, *, full_cov=False, q_sqrt=None, white=False, full_output_cov=None):
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N(0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case `f` represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).

    We assume R independent GPs, represented by the columns of f (and the
    first dimension of q_sqrt).

    :param Xnew: data matrix, size N x D. Evaluate the GP at these new points
    :param X: data points, size M x D.
    :param kern: GPflow kernel.
    :param f: data matrix, M x R, representing the function values at X,
        for R functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
    :param white: boolean of whether to use the whitened representation as
        described above.
    :return:
        - mean:     N x R
        - variance: N x R (full_cov = False), R x N x N (full_cov = True)
    """
    logger.debug("Conditional: Kernel")
    num_data = tf.shape(X)[-2]  # M
    Kmm = kern.K(X) + tf.eye(num_data, dtype=settings.float_type) * settings.numerics.jitter_level  #  [..., M, M]
    Kmn = kern.K(X, Xnew)  # [M, ..., N]

    if full_cov:
        Knn = kern.K(Xnew)  # [...,N,N]
    else:
        Knn = kern.Kdiag(Xnew)  # [...,N]

    mean, var = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    return mean, var  # N x R, N x R or R x N x N


# ----------------------------------------------------------------------------
############################ SAMPLE CONDITIONAL ##############################
# ----------------------------------------------------------------------------


@sample_conditional.register(object, object, Kernel, object)
@sample_conditional.register(object, InducingFeature, Kernel, object)
@name_scope("sample_conditional")
def _sample_conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False, num_samples=None):
    """
    `sample_conditional` will return a sample from the conditional distribution.
    In most cases this means calculating the conditional mean m and variance v and then
    returning m + sqrt(v) * eps, with eps ~ N(0, 1).
    However, for some combinations of Mok and Mof more efficient sampling routines exists.
    The dispatcher will make sure that we use the most efficient one.

    :return: samples, mean, cov
        samples has shape [num_samples, N, P] or [N, P] if num_samples is None
        mean and cov as for conditional()
    """
    if full_cov and full_output_cov:
        raise NotImplementedError("The combination of both full_cov and full_output_cov is not "
                                  "implemented for sample_conditional.")

    logger.debug("sample conditional: InducingFeature Kernel")
    mean, cov = conditional(Xnew, feat, kern, f, q_sqrt=q_sqrt, white=white,
                            full_cov=full_cov, full_output_cov=full_output_cov)
    if full_cov:
        # mean: [..., N, P]
        # cov: [..., P, N, N]
        mean_PN = tf.matrix_transpose(mean)  # [..., P, N]
        samples = _sample_mvn(mean_PN, cov, 'full', num_samples=num_samples)  # [..., (S), P, N]
        samples = tf.matrix_transpose(samples)  # [..., (S), P, N]

    else:
        # mean: [..., N, P]
        # cov: [..., N, P] or [..., N, P, P]
        cov_structure = "full" if full_output_cov else "diag"
        samples = _sample_mvn(mean, cov, cov_structure, num_samples=num_samples)  # [..., (S), P, N]

    return samples, mean, cov


# -----------------
# CONDITIONAL MATHS
# -----------------


@name_scope()
def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False):
    r"""
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: M x [...] x N
    :param Kmm: M x M
    :param Knn: [...] x N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    logger.debug("base conditional")
    # compute kernel stuff
    num_func = tf.shape(f)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f)[-2]

    # get the leadings dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat([tf.reshape(tf.range(1, K-1), [K-2]), # leading dims (...)
                      tf.reshape(0, [1]),  # [M]
                      tf.reshape(K-1, [1])], 0)  # [N]
    Kmn = tf.transpose(Kmn, perm)  # ... x M x N

    leading_dims = tf.shape(Kmn)[:-2]
    Lm = tf.cholesky(Kmm)  # [M,M]

    # Compute the projection matrix A
    Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [...,M,M]
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # [...,M,N]
    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)  # [...,N,N]
        cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -3), cov_shape)  # [...,R,N,N]
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [...,N]
        cov_shape = tf.concat([leading_dims, [num_func, N]], 0) # [...,R,N]
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [...,R,N]

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [...,M,R]
    f = tf.broadcast_to(f, f_shape)  # [...,M,R]
    fmean = tf.matmul(A, f, transpose_a=True)  # [...,N,R]

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = q_sqrt
            L = tf.broadcast_to(L, tf.concat([leading_dims, tf.shape(L)], 0))

            shape = tf.concat([leading_dims, [num_func, M, N]], 0)
            A_tiled = tf.broadcast_to(tf.expand_dims(A, -3), shape)
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # R x N

    if not full_cov:
        fvar = tf.matrix_transpose(fvar)  # N x R

    return fmean, fvar  # N x R, R x N x N or N x R

# ----------------------------------------------------------------------------
############################ UNCERTAIN CONDITIONAL ###########################
# ----------------------------------------------------------------------------

@name_scope()
def uncertain_conditional(Xnew_mu, Xnew_var, feat, kern, q_mu, q_sqrt, *,
                          mean_function=None, full_output_cov=False, full_cov=False, white=False):
    """
    Calculates the conditional for uncertain inputs Xnew, p(Xnew) = N(Xnew_mu, Xnew_var).
    See ``conditional`` documentation for further reference.

    :param Xnew_mu: mean of the inputs, size N x Din
    :param Xnew_var: covariance matrix of the inputs, size N x Din x Din
    :param feat: gpflow.InducingFeature object, only InducingPoints is supported
    :param kern: gpflow kernel object.
    :param q_mu: mean inducing points, size M x Dout
    :param q_sqrt: cholesky of the covariance matrix of the inducing points, size Dout x M x M
    :param full_output_cov: boolean wheter to compute covariance between output dimension.
                            Influences the shape of return value ``fvar``. Default is False
    :param white: boolean whether to use whitened representation. Default is False.

    :return fmean, fvar: mean and covariance of the conditional, size ``fmean`` is N x Dout,
            size ``fvar`` depends on ``full_output_cov``: if True ``f_var`` is N x Dout x Dout,
            if False then ``f_var`` is N x Dout
    """

    # TODO(VD): Tensorflow 1.7 doesn't support broadcasting in``tf.matmul`` and
    # ``tf.matrix_triangular_solve``. This is reported in issue 216.
    # As a temporary workaround, we are using ``tf.einsum`` for the matrix
    # multiplications and tiling in the triangular solves.
    # The code that should be used once the bug is resolved is added in comments.

    if not isinstance(feat, InducingPoints):
        raise NotImplementedError

    if full_cov:
        # TODO(VD): ``full_cov`` True would return a ``fvar`` of shape N x N x D x D,
        # encoding the covariance between input datapoints as well.
        # This is not implemented as this feature is only used for plotting purposes.
        raise NotImplementedError

    pXnew = Gaussian(Xnew_mu, Xnew_var)

    num_data = tf.shape(Xnew_mu)[0]  # number of new inputs (N)
    num_ind = tf.shape(q_mu)[0]  # number of inducing points (M)
    num_func = tf.shape(q_mu)[1]  # output dimension (D)

    q_sqrt_r = tf.matrix_band_part(q_sqrt, -1, 0)  # D x M x M

    eKuf = tf.transpose(expectation(pXnew, (kern, feat)))  # M x N (psi1)
    Kuu = features.Kuu(feat, kern, jitter=settings.jitter)  # M x M
    Luu = tf.cholesky(Kuu)  # M x M

    if not white:
        q_mu = tf.matrix_triangular_solve(Luu, q_mu, lower=True)
        Luu_tiled = tf.tile(Luu[None, :, :], [num_func, 1, 1])  # remove line once issue 216 is fixed
        q_sqrt_r = tf.matrix_triangular_solve(Luu_tiled, q_sqrt_r, lower=True)

    Li_eKuf = tf.matrix_triangular_solve(Luu, eKuf, lower=True)  # M x N
    fmean = tf.matmul(Li_eKuf, q_mu, transpose_a=True)

    eKff = expectation(pXnew, kern)  # N (psi0)
    eKuffu = expectation(pXnew, (kern, feat), (kern, feat))  # N x M x M (psi2)
    Luu_tiled = tf.tile(Luu[None, :, :], [num_data, 1, 1])  # remove this line, once issue 216 is fixed
    Li_eKuffu = tf.matrix_triangular_solve(Luu_tiled, eKuffu, lower=True)
    Li_eKuffu_Lit = tf.matrix_triangular_solve(Luu_tiled, tf.matrix_transpose(Li_eKuffu), lower=True)  # N x M x M
    cov = tf.matmul(q_sqrt_r, q_sqrt_r, transpose_b=True)  # D x M x M

    if mean_function is None or isinstance(mean_function, mean_functions.Zero):
        e_related_to_mean = tf.zeros((num_data, num_func, num_func), dtype=settings.float_type)
    else:
        # Update mean: \mu(x) + m(x)
        fmean = fmean + expectation(pXnew, mean_function)

        # Calculate: m(x) m(x)^T + m(x) \mu(x)^T + \mu(x) m(x)^T,
        # where m(x) is the mean_function and \mu(x) is fmean
        e_mean_mean = expectation(pXnew, mean_function, mean_function)  # N x D x D
        Lit_q_mu = tf.matrix_triangular_solve(Luu, q_mu, adjoint=True)
        e_mean_Kuf = expectation(pXnew, mean_function, (kern, feat))  # N x D x M
        # einsum isn't able to infer the rank of e_mean_Kuf, hence we explicitly set the rank of the tensor:
        e_mean_Kuf = tf.reshape(e_mean_Kuf, [num_data, num_func, num_ind])
        e_fmean_mean = tf.einsum("nqm,mz->nqz", e_mean_Kuf, Lit_q_mu)  # N x D x D
        e_related_to_mean = e_fmean_mean + tf.matrix_transpose(e_fmean_mean) + e_mean_mean

    if full_output_cov:
        fvar = (
                tf.matrix_diag(tf.tile((eKff - tf.trace(Li_eKuffu_Lit))[:, None], [1, num_func])) +
                tf.matrix_diag(tf.einsum("nij,dji->nd", Li_eKuffu_Lit, cov)) +
                # tf.matrix_diag(tf.trace(tf.matmul(Li_eKuffu_Lit, cov))) +
                tf.einsum("ig,nij,jh->ngh", q_mu, Li_eKuffu_Lit, q_mu) -
                # tf.matmul(q_mu, tf.matmul(Li_eKuffu_Lit, q_mu), transpose_a=True) -
                fmean[:, :, None] * fmean[:, None, :] +
                e_related_to_mean
        )
    else:
        fvar = (
                (eKff - tf.trace(Li_eKuffu_Lit))[:, None] +
                tf.einsum("nij,dji->nd", Li_eKuffu_Lit, cov) +
                tf.einsum("ig,nij,jg->ng", q_mu, Li_eKuffu_Lit, q_mu) -
                fmean ** 2 +
                tf.matrix_diag_part(e_related_to_mean)
        )

    return fmean, fvar


# ---------------------------------------------------------------
########################## HELPERS ##############################
# ---------------------------------------------------------------

def _sample_mvn(mean, cov, cov_structure=None, num_samples=None):
    """
    Returns a sample from a D-dimensional Multivariate Normal distribution
    :param mean: [..., N, D]
    :param cov: [..., N, D] or [..., N, D, D]
    :param cov_structure: "diag" or "full"
    - "diag": cov holds the diagonal elements of the covariance matrix
    - "full": cov holds the full covariance matrix (without jitter)
    :return: sample from the MVN of shape [..., (S), N, D], S = num_samples
    """
    mean_shape = tf.shape(mean)
    S = num_samples if num_samples is not None else 1
    D = mean_shape[-1]
    leading_dims = mean_shape[:-2]
    num_leading_dims = tf.size(leading_dims)

    if cov_structure == "diag":
        # mean: [..., N, D] and cov [..., N, D]
        with tf.control_dependencies([tf.assert_equal(tf.rank(mean), tf.rank(cov))]):
            eps_shape = tf.concat([leading_dims, [S], mean_shape[-2:]], 0)
            eps = tf.random_normal(eps_shape, dtype=settings.float_type)  # [..., S, N, D]
            samples = mean[..., None, :, :] + tf.sqrt(cov)[..., None, :, :] * eps  # [..., S, N, D]
    elif cov_structure == "full":
        # mean: [..., N, D] and cov [..., N, D, D]
        with tf.control_dependencies([tf.assert_equal(tf.rank(mean) + 1, tf.rank(cov))]):
            jittermat = (
                tf.eye(D, batch_shape=mean_shape[:-1], dtype=settings.float_type)
                * settings.jitter
            )  # [..., N, D, D]
            eps_shape = tf.concat([mean_shape, [S]], 0)
            eps = tf.random_normal(eps_shape, dtype=settings.float_type)  # [..., N, D, S]
            chol = tf.cholesky(cov + jittermat)  # [..., N, D, D]
            samples = mean[..., None] + tf.matmul(chol, eps)  # [..., N, D, S]
            samples = misc.leading_transpose(samples, [..., -1, -3, -2])  # [..., S, N, D]
    else:
        raise NotImplementedError  # pragma: no cover

    if num_samples is None:
        return samples[..., 0, :, :]  # [..., N, D]
    return samples  # [..., S, N, D]


def _expand_independent_outputs(fvar, full_cov, full_output_cov):
    """
    Reshapes fvar to the correct shape, specified by `full_cov` and `full_output_cov`.

    :param fvar: has shape N x P (full_cov = False) or P x N x N (full_cov = True).
    :return:
    1. full_cov: True and full_output_cov: True
       fvar N x P x N x P
    2. full_cov: True and full_output_cov: False
       fvar P x N x N
    3. full_cov: False and full_output_cov: True
       fvar N x P x P
    4. full_cov: False and full_output_cov: False
       fvar N x P
    """
    if full_cov and full_output_cov:
        fvar = tf.matrix_diag(tf.transpose(fvar))   # N x N x P x P
        fvar = tf.transpose(fvar, [0, 2, 1, 3])  # N x P x N x P
    if not full_cov and full_output_cov:
        fvar = tf.matrix_diag(fvar)   # N x P x P
    if full_cov and not full_output_cov:
        pass  # P x N x N
    if not full_cov and not full_output_cov:
        pass  # N x P

    return fvar


def _rollaxis_left(A, num_rolls):
    """ Roll the tensor `A` backwards `num_rolls` times """
    assert num_rolls > 0
    rank = tf.rank(A)
    perm = tf.concat([num_rolls + tf.range(rank - num_rolls), tf.range(num_rolls)], 0)
    return tf.transpose(A, perm)


def _rollaxis_right(A, num_rolls):
    """ Roll the tensor `A` forward `num_rolls` times """
    assert num_rolls > 0
    rank = tf.rank(A)
    perm = tf.concat([rank - num_rolls + tf.range(num_rolls), tf.range(rank - num_rolls)], 0)
    return tf.transpose(A, perm)
