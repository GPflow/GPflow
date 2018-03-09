import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

class settings:
    float_type = tf.float64


def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Given a variable g divided into disjoint subsets g_a and g_b, and distribution p and q such that
      p(g_b) = N(g_b;0,Kmm)
      p(g_a) = N(g_a;0,Knn)
      Cov[g_a,g_b] = Knm
    And
      q(g_b) = N(g_b;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g_a) = \int q(g_b) p(g_a|g_b)
    :param Kmn: M x N
    :param Kmm: M x M
    :param Knn: N x N (full_cov=True) or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular) Does not support q_diag, i.e. M x R
    :param white: bool
    :return: N x R  or N x N x R (full_cov=True)
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


def independent_latents_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None,  white=False):
    """
    Given P latent variables g1, ..., gP and disjoint subsets (g1_a, g1_b),...,(gP_a, gP_b)
      p(gi_b) = N(gi_b;0,Kmm(i))   i=1..P
      p(gi_a) = N(gi_a;0,Knn(i))
      Cov[gi_b,gi_a] = Knm(i)
    And
      q(gi_b) = N(gi_b;f(i),q_sqrt(i)*q_sqrt(i)^T)
    This method computes the mean and (co)variance of
      q(gi_a) = \int q(gi_b) p(gi_a|gi_b)

    This is intended to work for either
    * shared Kxxs across the latents ( i.e. Kmm : M x M, Knn : N x N, Knm : N x M )
    * separate Kxxs across the latents ( i.e. Kmm : P x M x M, Knn : P x N x N, Knm : P x N x M )

    The outputs be of shape
    * N x P  ,  N x P  (full_cov = False)
    * N x P  ,  N x N x P  (full_cov = True)

    Remarks:
    * Having P x P output makes no sense for independent latents (former full_cov_output irrelevant)
    * Whether one wants a full_cov (across the input of each gp) output for each gp should be reflected in the shape of Knn
                   _______________________________________________________________________
                  |______________Shared_______________|_______________Separate ___________|
                  |_____full_cov____|__not full_cov___|_____full_cov____|__not full_cov___|
                  |                 |                 |                 |                 |
    :param Kmn:   |    M x N        |     M x N       |    P x M x N    |    P x M x N    |
    :param Kmm:   |    M x M        |     M x N       |    P x M x M    |    P x M x M    |
    :param Knn:   |    N x N        |     N           |    P x N x N    |    P x N        |
                  |_______________________________________________________________________|
    :param f: data matrix, M x P
    :param q_sqrt:    P x M x M   # TODO  (1) no q_diag option ? (2) deal q_sqrt of shape (P x M x P x M)
    :return:    N x P  ,  N x P  or N x N x P (full_cov = True)
    """
    # TODO: Allow broadcasting over L if priors are shared?

    _, P = f.shape
    if Kmn.shape.ndims == 2: # shared Kxxs case
        N, M = Kmn.shape
        shared = True
        Kmn = tf.expand_dims(Kmn,0) # add 'P' dimension to the left
        Knn = tf.expand_dims(Knn,0) # add 'P' dimension to the left
        Kmm = tf.expand_dims(Kmm,0) # add 'P' dimension to the left
    elif Kmn.shape.ndims == 3: # separate Kxxs case
        _, N, M = Kmn.shape
        shared = False
    # if Knn.shape.ndims == 2:
    #     full_cov = True
    # else:
    #     full_cov = False


    Lm = tf.cholesky(Kmm)  #  P x M x M

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # P x M x M  *  P x M x N  ->  P x M x N

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.einsum('pmn,pms->pns',A, A)  # P x M x N, P x M x N -> P x N x N
    elif not full_cov:
        fvar = Knn - tf.einsum('pmn,pmn->pn',A, A)  # P x M x N, P x M x N -> P x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(Lm, A)  # P x M x M ,  P x M x N  ->  P x M x N

    fmean = tf.einsum('qmn,mp->np', A, f) # P x M x N , M x P -> N x P


    if q_sqrt is not None:
        Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # P x M x M
        if q_sqrt.shape.ndims == 3:
            LTA = tf.einsum('pmj,qjn->pmn', Lf, A) # P x M x M ,  P x M x N  ->  P x M x N
        else:
            raise NotImplementedError()

        if full_cov:
            fvar = fvar + tf.einsum('pmn,pmo->pno', LTA, LTA) # P x M x N , P x M x N -> P x N x N

        elif not full_cov:
            fvar = fvar + tf.einsum('pmn,pmn->pn', LTA, LTA) # P x M x N , P x M x N -> P x N



    return fmean, fvar


def fully_correlated_conditional(Kmn, Kmm, Knn, f, *, q_sqrt=None, white=False):
    """
    Given P latent variables g1, ..., gP and disjoint subsets (g1_a, g1_b),...,(gP_a, gP_b)
      p(g_b) = N(g_b;0,Kmm)
      p(g_a) = N(g_a;0,Knn)
      Cov[g_a,g_b] = Knm
    And
      q(g_b) = N(g_b;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g_a) = \int q(g_b) p(g_a|g_b)

    We have
    :param Kmn:    P x M x P x N
    :param Kmm:    P x M x P x M
    :param Knn:    P x N x P x N
    :param f: data matrix, M x P
    :param q_sqrt:    P x M x P x M
    :return:    N x P,  P x N x P x N
    """

    P,M,_,N = Kmn.shape

    # TODO: lots of reshaping going on. Check ordering of dimensions is correct

    # TODO implement a block cholesky as in https://scicomp.stackexchange.com/questions/5050/cholesky-factorization-of-block-matrices
    Kmmr = tf.reshape(Kmm, (P*M, P*M)) # PM x PM
    Lmr = tf.cholesky(Kmmr) # PM x PM

    # Compute the projection matrix A
    Kmnr = tf.reshape(Kmn, (M*P, N*P))  # PM x PN
    Ar = tf.matrix_triangular_solve(Lmr, Kmnr, lower=True)  # PM x PM , PM x PN -> PM x PN

    # compute the covariance due to the conditioning
    Knnr = tf.reshape(Knn, (N*P, N*P))
    fvarr = Knnr - tf.matmul(Ar, Ar, transpose_a=True)  # PM x PN, PM x PN -> PN x PN

    # another backsubstitution in the unwhitened case
    if not white:
        Ar = tf.matrix_triangular_solve(Lmr, Ar)  # PM x PM, PM x PN -> PM x PN

    fr = tf.reshape(f, (P*M,1)) # M x P -> PM
    fmeanr = tf.matmul(fr, Ar, transpose_a=True)  # PM, PM x PN  ->  PN

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 4:
            q_sqrtr = tf.reshape(q_sqrt, (M*P,M*P)) # PM x PM
            Lfr = tf.matrix_band_part(q_sqrtr, -1, 0)  # PM x PM
            LTA = tf.matmul(Lfr, Ar, transpose_a=True)  # PM x PM , PM x PN -> PM x PN
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))

        fvarr = fvarr + tf.matmul(LTA, LTA, transpose_a=True)  # PM x PN, PM x PN -> PN x PN


    return tf.reshape(fmeanr, (N, P)),\
           tf.reshape(fvarr, (P, N, P, N))


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
Kmn = tf.constant( np.eye(N)[:M,:], dtype=settings.float_type)
Kmm = tf.constant( np.eye(M), dtype=settings.float_type)
Knn = tf.constant( np.ones((N,N)), dtype=settings.float_type)
f = tf.constant( np.ones((M,R)), dtype=settings.float_type)
full_cov = True

q_sqrt = tf.constant( np.array([np.eye(M)
                              for _ in range(R)]), dtype=settings.float_type)
q_sqrt = None
#print(q_sqrt.shape)

m,v = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=False)

with tf.Session() as sess:
    print( sess.run(tf.shape(v)) )

'''
# ===========================================
'''

N = 100
M = 10
P = 2






def make_test_input(full_cov,shared):

    if full_cov and shared:

        Kmn = tf.constant( np.ones((M,N)), dtype=settings.float_type)
        Kmm = tf.constant( np.eye(M) , dtype=settings.float_type)
        Knn = tf.constant( np.ones((N,N)), dtype=settings.float_type)
        f = tf.constant( np.ones((M,P)), dtype=settings.float_type)
        q_sqrt = tf.constant( np.array([np.eye(M)
                                       for _ in range(P)]), dtype=settings.float_type)

    elif not full_cov and shared:

        Kmn = tf.constant( np.ones((M,N)), dtype=settings.float_type)
        Kmm = tf.constant( np.eye(M) , dtype=settings.float_type)
        Knn = tf.constant( np.ones((N,)), dtype=settings.float_type)
        f = tf.constant( np.ones((M,P)), dtype=settings.float_type)
        q_sqrt = tf.constant( np.array([np.eye(M)
                                       for _ in range(P)]), dtype=settings.float_type)


    elif full_cov and not shared:
        Kmn = tf.constant( np.ones((P,M,N)), dtype=settings.float_type)
        Kmm = tf.constant( np.array([np.eye(M)  for _ in range(P)]), dtype=settings.float_type)
        Knn = tf.constant( np.ones((P,N,N)), dtype=settings.float_type)
        f = tf.constant( np.ones((M,P)), dtype=settings.float_type)
        q_sqrt = tf.constant( np.array([np.eye(M)
                                       for _ in range(P)]), dtype=settings.float_type)
    elif not full_cov and not shared:

        Kmn = tf.constant( np.ones((P,M,N)), dtype=settings.float_type)
        Kmm = tf.constant( np.array([np.eye(M)  for _ in range(P)]), dtype=settings.float_type)
        Knn = tf.constant( np.ones((P,N)), dtype=settings.float_type)
        f = tf.constant( np.ones((M,P)), dtype=settings.float_type)
        q_sqrt = tf.constant( np.array([np.eye(M)
                                       for _ in range(P)]), dtype=settings.float_type)



    return Kmn, Kmm, Knn, f, q_sqrt


with tf.Session() as sess:

    for full_cov in [True,False]:
        for shared in [True,False]:
            print('full_cov:',full_cov,', shared:',shared)
            Kmn, Kmm, Knn, f, q_sqrt = make_test_input(full_cov,shared)
            m,v = independent_latents_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=False)

            print( sess.run(tf.shape(m)) )
            print( sess.run(tf.shape(v)) )


'''


"""
    :param Kmn:    P x M x P x N
    :param Kmm:    P x M x P x M
    :param Knn:    P x N x P x N
    :param f: data matrix, M x P
    :param q_sqrt:    P x M x P x M
    :return:    N x P,  P x N x P x N
    
"""
P,M,N=3,10,100
Kmn = tf.constant( np.ones((P,M,P,N)), dtype=settings.float_type)

Kmm_np  = np.zeros((1,1,M,M))
Kmm_np[0,0,:,:] = np.eye(M)
Kmm_np = np.tile(Kmm_np,[P,P,1,1])
Kmm_np = np.reshape(Kmm_np,(P,M,P,M))

Kmm = tf.constant( Kmm_np, dtype=settings.float_type)

Knn = tf.constant( np.ones((P,N,P,N)), dtype=settings.float_type)
q_sqrt = tf.constant( Kmm_np, dtype=settings.float_type)
f = tf.constant( np.ones((M,P)), dtype=settings.float_type)


with tf.Session() as sess:

    m,v = fully_correlated_conditional(Kmn, Kmm, Knn, f,  q_sqrt=None, white=False)
    print( sess.run(tf.shape(m)) )
    print( sess.run(tf.shape(v)) )

