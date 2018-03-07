import tensorflow as tf
import numpy as np

class settings:
    float_type = tf.float64


def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Given a variable g divided into disjoint subsets g_a and g_b, and distribution p and q such that
      p(g_b) = N(g_b;0,Kmm)
      p(g_a) = N(g_a;0,Knn)
      p(g_a|g_b) = N(g_a;0,Knm)
    And
      q(g_b) = N(g_b;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g_a) = \int q(g_b) p(g_a|g_b)
    :param Kmn: M x N
    :param Kmm: M x M
    :param Knn: N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or N x N x R
    """
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # R
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
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # R x N x N or R x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # R x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N
    fvar = tf.transpose(fvar)  # N x R or N x N x R

    return fmean, fvar


def independent_latents_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_cov_output=False, q_sqrt=None,
                                    white=False):
    """
    Given P latent variables g1, ..., gP and disjoint subsets (g1_a, g1_b),...,(gP_a, gP_b)
      p(gi_b) = N(gi_b;0,Kmm(i))
      p(gi_a) = N(gi_a;0,Knn(i))
      p(gi_a|gi_b) = N(gi_a;0,Knm(i))
    And
      q(gi_b) = N(gi_b;f(i),q_sqrt(i)*q_sqrt(i)^T)
    This method computes the mean and (co)variance of
      q(gi_a) = \int q(gi_b) p(gi_a|gi_b)
    for i in [1..P]


    :param Kmn: M x P x N x P
    :param Kmm: P x M x M
    :param Knn: N x P  or  N x N  or  P x N x N  or  N x P x N x P
    :param f: data matrix, M x P
    :param q_sqrt: P x M x M  or  M x P
    :return: N x P  ,  N x P x P
    """
    # TODO: Allow broadcasting over L if priors are shared?
    # TODO: Change Kmn to be L x M x N x P? Saves a transpose...
    M, P, N, _ = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]

    Lm = tf.cholesky(Kmm)  # L x M x M

    # Compute the projection matrix A
    Kmn = tf.reshape(tf.transpose(Kmn, (1, 0, 2, 3)), (P, M, N * P))
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # P x M x M  *  P x M x NP  ->  P x M x NP
    Ar = tf.reshape(A, (P, M, N, P))

    # compute the covariance due to the conditioning
    if full_cov and full_cov_output:
        fvar = Knn - tf.tensordot(Ar, Ar, [[0, 1], [0, 1]])  # N x P x N x P
    elif full_cov and not full_cov_output:
        At = tf.reshape(tf.transpose(Ar), (P, N, M * P))  # P x N x MP
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # P x N x N
    elif not full_cov and full_cov_output:
        At = tf.reshape(tf.transpose(Ar, [2, 3, 1, 0]), (N, P, M * P))  # N x P x MP
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # N x P x P
    elif not full_cov and not full_cov_output:
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0, 1]), (N, P))  # Knn: N x P

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(Lm, A)  # P x M x M  *  P x M x NP  ->  P x M x NP
        Ar = tf.reshape(A, (P, M, N, P))


    fmean = tf.tensordot(Ar, f, [[0, 1], [0, 1]])  # N x P

    if q_sqrt is not None:
        Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # P x M x M
        if q_sqrt.shape.ndims == 3:
            LTA = tf.matmul(Lf, A, transpose_a=True)  # P x M x M  *  P x M x NP  ->  P x M x NP
        else:
            raise NotImplementedError()

        if full_cov and full_cov_output:
            LTAr = tf.reshape(LTA, (P * M, N * P))
            fvar = fvar + tf.reshape(tf.matmul(LTAr, LTAr, transpose_a=True), (N, P, N, P))
        elif full_cov and not full_cov_output:
            LTAr = tf.transpose(tf.reshape(LTA, (P * M, N, P)), [0, 3, 1, 2])  # P x PM x N
            fvar = fvar + tf.matmul(LTAr, LTAr, transpose_a=True)  # P x N x N
        elif not full_cov and full_cov_output:
            LTAr = tf.transpose(tf.reshape(LTA, (P * M, N, P)), [1, 0, 2])  # N x PM x P
            fvar = fvar + tf.matmul(LTAr, LTAr, transpose_a=True)  # N x P x P
        elif not full_cov and not full_cov_output:
            fvar = fvar + tf.reshape(tf.reduce_sum(tf.square(LTA), (0, 1)), (N, P))
    return fmean, fvar



# ==========================================
'''
N = 100
M = 10
R = 2


# Non full cov
Kmn = tf.constant( np.eye(N)[:M,:], dtype=settings.float_type)
Kmm = tf.constant( np.eye(M), dtype=settings.float_type)
Knn = tf.constant( np.ones((N,)), dtype=settings.float_type)
f = tf.constant( np.ones((M,R)), dtype=settings.float_type)
full_cov = False

# full cov
#Kmn = tf.constant( np.eye(N)[:M,:], dtype=float_type)
#Kmm = tf.constant( np.eye(M), dtype=float_type)
#Knn = tf.constant( np.ones((N,N)), dtype=float_type)
#f = tf.constant( np.ones((M,R)), dtype=float_type)
#full_cov = True

#q_sqrt = tf.constant( np.array([np.eye(M)
#                               for _ in range(R)]), dtype=float_type)
q_sqrt = None
#print(q_sqrt.shape)

m,v = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=False)

with tf.Session() as sess:
    print( sess.run(tf.shape(v)) )

'''
# ===========================================
'''
   :param Kmn: M x P x N x P
    :param Kmm: P x M x M
    :param Knn: N x P  or  N x N  or  P x N x N  or  N x P x N x P
    :param f: data matrix, M x P
    :param q_sqrt: P x M x M  or  M x P
    :return: N x P  ,  N x P x P
'''
N = 100
M = 10
P = 2


# Non full cov
Kmn = tf.constant( np.ones((M,P,N,P)), dtype=settings.float_type)
Kmm = tf.constant( np.array([np.eye(M) for _ in range(P)]), dtype=settings.float_type)

Knn = tf.constant( np.ones((N,P)), dtype=settings.float_type)
#Knn = tf.constant( np.ones((N,N)), dtype=settings.float_type)
#Knn = tf.constant( np.ones((P,N,N)), dtype=settings.float_type)
#Knn = tf.constant( np.ones((N,P,N,P)), dtype=settings.float_type)

f = tf.constant( np.ones((M,P)), dtype=settings.float_type)
full_cov = False
full_cov_output = True


#q_sqrt = tf.constant( np.array([np.eye(M)
#                               for _ in range(R)]), dtype=float_type)
q_sqrt = None
#print(q_sqrt.shape)

m,v = independent_latents_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,full_cov_output=full_cov_output, q_sqrt=q_sqrt, white=False)

with tf.Session() as sess:
    print( sess.run(tf.shape(m)) )
    print( sess.run(tf.shape(v)) )
