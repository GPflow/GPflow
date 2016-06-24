import tensorflow as tf
from .tf_hacks import eye


def gauss_kl_white(q_mu, q_sqrt, num_latent):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, I)

    We assume num_latent independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and the last dim of q_sqrt).
    """
    KL = 0.5 * tf.reduce_sum(tf.square(q_mu))  # Mahalanobis term
    KL += -0.5 * tf.cast(tf.shape(q_sqrt)[0] * num_latent, tf.float64)
    for d in range(num_latent):
        Lq = tf.batch_matrix_band_part(q_sqrt[:, :, d], -1, 0)
        # Log determinant of q covariance:
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.square(tf.diag_part(Lq))))
        KL += 0.5 * tf.reduce_sum(tf.square(Lq))  # Trace term.
    return KL


def gauss_kl_white_diag(q_mu, q_sqrt, num_latent):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, I)

    We assume num_latent independent distributions, given by the columns of
    q_mu and q_sqrt

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each column represents the diagonal of a square-root
        matrix of the covariance.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and q_sqrt).
    """

    KL = 0.5 * tf.reduce_sum(tf.square(q_mu))  # Mahalanobis term
    KL += -0.5 * tf.cast(tf.shape(q_sqrt)[0] * num_latent, tf.float64)
    KL += -0.5 * tf.reduce_sum(tf.log(tf.square(q_sqrt)))  # Log-det of q-cov
    KL += 0.5 * tf.reduce_sum(tf.square(q_sqrt))  # Trace term
    return KL


def gauss_kl_diag(q_mu, q_sqrt, K,  num_latent):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume num_latent independent distributions, given by the columns of
    q_mu and q_sqrt.

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each column represents the diagonal of a square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and q_sqrt).
    """
    L = tf.cholesky(K)
    alpha = tf.matrix_triangular_solve(L, q_mu, lower=True)
    KL = 0.5 * tf.reduce_sum(tf.square(alpha))  # Mahalanobis term.
    KL += num_latent * 0.5 * tf.reduce_sum(
        tf.log(tf.square(tf.diag_part(L))))  # Prior log-det term.
    KL += -0.5 * tf.cast(tf.shape(q_sqrt)[0] * num_latent, tf.float64)
    KL += -0.5 * tf.reduce_sum(tf.log(tf.square(q_sqrt)))  # Log-det of q-cov
    L_inv = tf.matrix_triangular_solve(L, eye(tf.shape(L)[0]), lower=True)
    K_inv = tf.matrix_triangular_solve(tf.transpose(L), L_inv, lower=False)
    KL += 0.5 * tf.reduce_sum(tf.expand_dims(tf.diag_part(K_inv), 1)
                              * tf.square(q_sqrt))  # Trace term.
    return KL


def gauss_kl(q_mu, q_sqrt, K, num_latent):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume num_latent independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean.

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.

    num_latent is an integer: the number of independent distributions (equal to
        the columns of q_mu and the last dim of q_sqrt).
    """
    L = tf.cholesky(K)
    alpha = tf.matrix_triangular_solve(L, q_mu, lower=True)
    KL = 0.5 * tf.reduce_sum(tf.square(alpha))  # Mahalanobis term.
    KL += num_latent * 0.5 * tf.reduce_sum(
        tf.log(tf.square(tf.diag_part(L))))  # Prior log-det term.
    KL += -0.5 * tf.cast(tf.shape(q_sqrt)[0] * num_latent, tf.float64)
    for d in range(num_latent):
        Lq = tf.batch_matrix_band_part(q_sqrt[:, :, d], -1, 0)
        # Log determinant of q covariance:
        KL += -0.5*tf.reduce_sum(tf.log(tf.square(tf.diag_part(Lq))))
        LiLq = tf.matrix_triangular_solve(L, Lq, lower=True)
        KL += 0.5 * tf.reduce_sum(tf.square(LiLq))  # Trace term
    return KL
