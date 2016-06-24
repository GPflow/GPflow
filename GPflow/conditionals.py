from .tf_hacks import eye
import tensorflow as tf


def conditional(Xnew, X, kern, f, num_columns,
                full_cov=False, q_sqrt=None, whiten=False):
    """
    Given F, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there my be Gaussian uncertainty about F as represented by
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
     - f is a data matrix, M x K, representing the function values at X.
     - num_columns is an integer number of columns in the f matrix (must match
       q_sqrt's last dimension)
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

    # compute kernel stuff
    num_data = tf.shape(X)[0]
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + eye(num_data) * 1e-6
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
        fvar = tf.tile(tf.expand_dims(fvar, 2), [1, 1, num_columns])
    else:
        fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
        fvar = tf.tile(tf.expand_dims(fvar, 1), [1, num_columns])

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(tf.transpose(A), f)

    # add extra projected variance from q(f) if needed
    if q_sqrt is not None:
        projected_var = []
        for d in range(num_columns):
            if q_sqrt.get_shape().ndims == 2:
                LTA = A * q_sqrt[:, d:d + 1]
            elif q_sqrt.get_shape().ndims == 3:
                L = tf.batch_matrix_band_part(q_sqrt[:, :, d], -1, 0)
                LTA = tf.matmul(tf.transpose(L), A)
            else:  # pragma no cover
                raise ValueError("Bad dimension for q_sqrt: %s" %
                                 str(q_sqrt.get_shape().ndims))
            if full_cov:
                projected_var.append(tf.matmul(tf.transpose(LTA), LTA))
            else:
                projected_var.append(tf.reduce_sum(tf.square(LTA), 0))
        fvar = fvar + tf.transpose(tf.pack(projected_var))

    return fmean, fvar


import warnings


def gp_predict(Xnew, X, kern, F, full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, F, num_columns=1,
                       full_cov=full_cov, q_sqrt=None, whiten=False)


def gaussian_gp_predict(Xnew, X, kern, q_mu, q_sqrt, num_columns,
                        full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, q_mu, num_columns=num_columns,
                       full_cov=full_cov, q_sqrt=q_sqrt, whiten=False)


def gaussian_gp_predict_whitened(Xnew, X, kern, q_mu, q_sqrt, num_columns,
                                 full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, q_mu, num_columns=num_columns,
                       full_cov=full_cov, q_sqrt=q_sqrt, whiten=True)


def gp_predict_whitened(Xnew, X, kern, V, full_cov=False):
    warnings.warn('gp_predict is deprecated: use conditonal(...) instead',
                  DeprecationWarning)
    return conditional(Xnew, X, kern, V, num_columns=1,
                       full_cov=full_cov, q_sqrt=None, whiten=True)
