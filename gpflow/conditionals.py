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

from . import settings, mean_functions
from .decors import name_scope
from .expectations import expectation
from .features import InducingPoints
from .probability_distributions import Gaussian


@name_scope()
def conditional(Xnew, X, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
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

    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

    :param Xnew: data matrix, size N x D.
    :param X: data points, size M x D.
    :param kern: GPflow kernel.
    :param f: data matrix, M x K, representing the function values at X,
        for K functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x K or K x M x M.
    :param white: boolean of whether to use the whitened representation as
        described above.

    :return: two element tuple with conditional mean and variance.
    """
    num_data = tf.shape(X)[0]  # M
    Kmm = kern.K(X) + tf.eye(num_data, dtype=settings.float_type) * settings.numerics.jitter_level
    Kmn = kern.K(X, Xnew)
    if not full_cov_output:
        Knn = kern.K(Xnew) if full_cov else kern.Kdiag(Xnew)
    # elif issubclass(kern, IMultiKernel):
    else:
        Knn = kern.K(Xnew, full_cov_output=full_cov_output) if full_cov else \
            kern.Kdiag(Xnew, full_cov_output=full_cov_output)

    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, full_cov_output=full_cov_output, q_sqrt=q_sqrt,
                            white=white)


@name_scope()
def feature_conditional(Xnew, feat, kern, f, *, full_cov=False, q_sqrt=None, white=False):
    Kmm = feat.Kuu(kern, jitter=settings.numerics.jitter_level)
    Kmn = feat.Kuf(kern, Xnew)
    if full_cov:
        Knn = kern.K(Xnew)
    else:
        Knn = kern.Kdiag(Xnew)
    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)


@name_scope()
def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    This function handles conditioning of single and multiple output GPs in
    various situations.
     - Single output
       Kmn: M x N
       Kmm: M x M
       f  : M x 1
     - Multiple independent outputs, shared kernels
       Kmn: M x N
       Kmm: M x M
       f  : M x K
     - Multiple independent outputs, multiple kernels
       Kmn: K x M x N
       Kmm: K x M x M
       f  : M x K
     - Multiple dependent outputs
       Kmn: M x N x K
       Kmm: M x M
       f  : M x 1
    :param Kmn: M x N  or  K x M x N  TODO: Should be M x N x K
    :param Kmm: M x M  or  K x M x M
    :param Knn: N  or  N x K  or  N x N  or K x N x N
    :param f: data matrix, M x K, representing the function values at X,
        for K functions.
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt:
    :param white:
    :return:
    """

    if full_cov_output:
        raise NotImplementedError()

    # compute kernel stuff
    num_func = tf.shape(f)[1]  # K
    Lm = tf.cholesky(Kmm)  # M x M  or  L x M x M
    if Lm.shape.ndims == 2:
        Lm = Lm[None, :, :]

    L = None if Kmm.shape.ndims == 2 else tf.shape(Kmm)[0]
    M = tf.shape(Kmm)[1]
    L_is_K = False
    Kmn_shape = tf.shape(Kmn)
    if Kmn.shape.ndims == 2:
        # Either single output, or multi-output with identical kernels *and* different inducing variables. f: M x K
        K = tf.shape(f)[1]
        N = tf.shape(Kmn)[1]
        Kmn = Kmn[None, :, :]
        L_is_K = True
    elif Kmn.shape.ndims == 3:  # Kmn: K x M x N
        # Multi-output with different kernels *and* different inducing variables. f: M x K
        K, N = Kmn_shape[0], Kmn_shape[2]
        L_is_K = True
    elif Kmn.shape.ndims == 4:  # Kmn: L x M x N x K
        # Multi-output with multi-output inducing variables (full whammy). f: M x L
        N, K = tf.shape(Kmn)[2:]
        Kmn = tf.reshape(Kmn, (L, M, N * K))
    else:
        raise NotImplementedError()

    # Compute the projection matrix A
    # Lm: L x M x M     Kmn: L x M x NK
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # L1 x M x NK

    # compute the covariance due to the conditioning
    if full_cov:
        if full_cov_output:
            Ar = tf.reshape(A, (-1, N * K))
            fvar = Knn - tf.matmul(Ar, Ar, transpose_a=True)  # NK x NK
        else:
            # Knn: K x N x N
            Ar = tf.transpose(tf.reshape(A, (-1, N, K)))  # K x N x ML
            fvar = Knn - tf.matmul(Ar, Ar, transpose_b=True)  # K x N x N
    else:
        if full_cov_output:
            raise NotImplementedError()
        else:
            if L_is_K:
                fvar = tf.transpose(Knn) - tf.reduce_sum(tf.square(A), 1)  # K1 x N

    # another backsubstitution in the unwhitened case
    if not white:
        # if A.shape[0] == K, then Lm.shape[0] == K as well
        A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False)  # M x NK  or  L x M x NK

    if L_is_K:
        # In this case, f is of size M x K
        # A: LK1 x M x N  f.T: K x M x 1 -> K x N x 1[:, :, 0].T -> N x K
        A_tiled = tf.tile(A, tf.stack([num_func // tf.shape(A)[0], 1, 1]))
        fmean = tf.transpose(tf.matmul(A_tiled, tf.transpose(f)[:, :, None], transpose_a=True)[:, :, 0])
    else:
        raise NotImplementedError()

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # K x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # K x M x M
            if L_is_K:
                A_tiled = tf.tile(A, tf.stack([num_func // tf.shape(A)[0], 1, 1]))
                LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar


@name_scope()
def uncertain_conditional(Xnew_mu, Xnew_var, feat, kern, q_mu, q_sqrt, *,
                          mean_function=None, full_cov_output=False, full_cov=False, white=False):
    """
    Calculates the conditional for uncertain inputs Xnew, p(Xnew) = N(Xnew_mu, Xnew_var).
    See ``conditional`` documentation for further reference.

    :param Xnew_mu: mean of the inputs, size N x Din
    :param Xnew_var: covariance matrix of the inputs, size N x Din x Din
    :param feat: gpflow.InducingFeature object, only InducingPoints is supported
    :param kern: gpflow kernel or ekernel object.
    :param q_mu: mean inducing points, size M x Dout
    :param q_sqrt: cholesky of the covariance matrix of the inducing points, size Dout x M x M
    :param full_cov_output: boolean wheter to compute covariance between output dimension.
                            Influences the shape of return value ``fvar``. Default is False
    :param white: boolean whether to use whitened representation. Default is False.

    :return fmean, fvar: mean and covariance of the conditional, size ``fmean`` is N x Dout,
            size ``fvar`` depends on ``full_cov_output``: if True ``f_var`` is N x Dout x Dout,
            if False then ``f_var`` is N x Dout
    """

    # TODO: Tensorflow 1.4 doesn't support broadcasting in``tf.matmul`` and
    # ``tf.matrix_triangular_solve``. This is reported in issue 216.
    # As a temporary workaround, we are using ``tf.einsum`` for the matrix
    # multiplications and tiling in the triangular solves.
    # The code that should be used once the bug is resolved is added in comments.

    if not isinstance(feat, InducingPoints):
        raise NotImplementedError

    if full_cov:
        # TODO: ``full_cov`` True would return a ``fvar`` of shape N x N x D x D,
        # encoding the covariance between input datapoints as well.
        # This is not implemented as this feature is only used for plotting purposes.
        raise NotImplementedError

    pXnew = Gaussian(Xnew_mu, Xnew_var)

    num_data = tf.shape(Xnew_mu)[0]  # number of new inputs (N)
    num_ind = tf.shape(q_mu)[0]  # number of inducing points (M)
    num_func = tf.shape(q_mu)[1]  # output dimension (D)

    q_sqrt_r = tf.matrix_band_part(q_sqrt, -1, 0)  # D x M x M

    eKuf = tf.transpose(expectation(pXnew, (feat, kern)))  # M x N (psi1)
    Kuu = feat.Kuu(kern, jitter=settings.numerics.jitter_level)  # M x M
    Luu = tf.cholesky(Kuu)  # M x M

    if not white:
        q_mu = tf.matrix_triangular_solve(Luu, q_mu, lower=True)
        Luu_tiled = tf.tile(Luu[None, :, :], [num_func, 1, 1])  # remove line once issue 216 is fixed
        q_sqrt_r = tf.matrix_triangular_solve(Luu_tiled, q_sqrt_r, lower=True)

    Li_eKuf = tf.matrix_triangular_solve(Luu, eKuf, lower=True)  # M x N
    fmean = tf.matmul(Li_eKuf, q_mu, transpose_a=True)

    eKff = expectation(pXnew, kern)  # N (psi0)
    eKuffu = expectation(pXnew, (feat, kern), (feat, kern))  # N x M x M (psi2)
    Luu_tiled = tf.tile(Luu[None, :, :], [num_data, 1, 1])  # remove this line, once issue 216 is fixed
    Li_eKuffu_Lit = tf.matrix_triangular_solve(Luu_tiled, tf.matrix_transpose(eKuffu), lower=True)
    Li_eKuffu_Lit = tf.matrix_triangular_solve(Luu_tiled, tf.matrix_transpose(Li_eKuffu_Lit), lower=True)  # N x M x M
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
        e_mean_Kuf = expectation(pXnew, mean_function, (feat, kern))  # N x D x M
        # einsum isn't able to infer the rank of e_mean_Kuf, hence we explicitly set the rank of the tensor:
        e_mean_Kuf = tf.reshape(e_mean_Kuf, [num_data, num_func, num_ind])
        e_fmean_mean = tf.einsum("nqm,mz->nqz", e_mean_Kuf, Lit_q_mu)  # N x D x D
        e_related_to_mean = e_fmean_mean + tf.matrix_transpose(e_fmean_mean) + e_mean_mean

    if full_cov_output:
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
