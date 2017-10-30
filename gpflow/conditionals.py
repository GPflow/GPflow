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


import warnings

import tensorflow as tf
from gpflow import settings
from gpflow.decors import name_scope


@name_scope()
def uncertain_conditional(Xnew_mu, Xnew_var, feat, kern, f, q_sqrt=None, whiten=False):
    assert whiten
    # Xnew_mu: N x D
    # Xnew_var: N x D x D
    # f: M x D
    eKuf = tf.transpose(feat.eKfu(kern, Xnew_mu, Xnew_var))  # M x N
    Kuu = feat.Kuu(kern, jitter=settings.numerics.jitter_level)

    num_func = tf.shape(f)[1]
    Lm = tf.cholesky(Kuu)

    A = tf.matrix_triangular_solve(Lm, eKuf, lower=True)

    eKff = kern.eKdiag(Xnew_mu, Xnew_var)
    eKuffu = feat.eKufKfu(kern, Xnew_mu, Xnew_var)  # NxMxM
    Li_eKuffu_Lit = tf.matrix_triangular_solve(Lm[None, :, :], tf.transpose(eKuffu, (0, 2, 1)), lower=True,
                                               adjoint=True)
    Li_eKuffu_Lit = tf.matrix_triangular_solve(Lm[None, :, :], tf.transpose(Li_eKuffu_Lit, (0, 2, 1)))
    print(Li_eKuffu_Lit.get_shape())
    q_sqrt_r = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)
    cov = tf.matmul(q_sqrt_r, q_sqrt_r, transpose_b=True)
    fvar = (
        tf.matrix_diag(eKff - tf.trace(Li_eKuffu_Lit))
        # tf.matrix_diag(tf.trace(tf.matmul(Li_eKuffu_Lit[:, None, :, :], cov[None, :, :, :]))) +
        # tf.matmul(f[None, :, :], tf.matmul(Li_eKuffu_Lit, f[None, :, :]), transpose_a=True)
    )

    if not whiten:
        A = tf.matrix_triangular_solve(Lm, A, lower=True, adjoint=True)  # Now A = Kmm^-1 eKuf

    fmean = tf.matmul(A, f, transpose_a=True)

    return fmean, tf.Print(fvar, [eKff, tf.trace(Li_eKuffu_Lit)])


@name_scope()
def conditional(Xnew, X, kern, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Given F, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about F as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

     - Xnew is a data matrix, size N x D
     - X are data points, size M x D
     - kern is a GPflow kernel
     - f is a data matrix, M x K, representing the function values at X, for K functions.
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x K or M x M x K
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.

    These functions are now considered deprecated, subsumed into this one:
        gp_predict
        gaussian_gp_predict
        gp_predict_whitened
        gaussian_gp_predict_whitened

    """
    num_data = tf.shape(X)[0]  # M
    Kmm = kern.K(X) + tf.eye(num_data, dtype=settings.tf_float) * settings.numerics.jitter_level
    Kmn = kern.K(X, Xnew)
    if full_cov:
        Knn = kern.K(Xnew)
    else:
        Knn = kern.Kdiag(Xnew)
    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, whiten=whiten)


@name_scope()
def feature_conditional(Xnew, feat, kern, f, full_cov=False, q_sqrt=None, whiten=False):
    Kmm = feat.Kuu(kern, jitter=settings.numerics.jitter_level)
    Kmn = feat.Kuf(kern, Xnew)
    if full_cov:
        Knn = kern.K(Xnew)
    else:
        Knn = kern.Kdiag(Xnew)
    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, whiten=whiten)


@name_scope()
def base_conditional(Kmn, Kmm, Knn, f, full_cov=False, q_sqrt=None, whiten=False):
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # K
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)
        shape = tf.stack([num_func, 1, 1])
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # K x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # K x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
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


def gp_predict(Xnew, X, kern, F, full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, F,
                       full_cov=full_cov, q_sqrt=None, whiten=False)


def gaussian_gp_predict(Xnew, X, kern, q_mu, q_sqrt, num_columns,
                        full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, q_mu,
                       full_cov=full_cov, q_sqrt=q_sqrt, whiten=False)


def gaussian_gp_predict_whitened(Xnew, X, kern, q_mu, q_sqrt, num_columns,
                                 full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, q_mu,
                       full_cov=full_cov, q_sqrt=q_sqrt, whiten=True)


def gp_predict_whitened(Xnew, X, kern, V, full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, V,
                       full_cov=full_cov, q_sqrt=None, whiten=True)
