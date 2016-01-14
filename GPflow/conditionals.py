import numpy as np
from tf_hacks import eye
import tensorflow as tf

"""
This file contains functions which compute tensorflow expressions for GP prediction
equations. This enables much code re-use, because these expressions appear
frequently. 
"""

def gp_predict(Xnew, X, kern, F):
    """
    Given F, representing the GP at the points X, produce the mean and variance
    of the GP at the points Xnew.

    We assume K independent GPs, represented by the columns of F 
    . 
    This function computes the Gaussian conditional
        p(F* | F) 

    Xnew is a data matrix, size N x D
    X are inducing points, size M x D
    F are function values , size M x K

    See also:
        gp_predict_whitened -- where F is rotated into V (F = LV)
        gaussian_gp_predict -- similar, but with uncertainty in F

    """
 
    #compute kernel stuff
    num_data = X.shape[0]
    Kdiag = kern.Kdiag(Xnew)
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + eye(num_data)*1e-4
    Lm = tf.cholesky(Kmm)

    #this is O(N M^2)
    A = tf.user_ops.triangular_solve(Lm, Kmn, 'lower')
    B = tf.user_ops.triangular_solve(tf.transpose(Lm), A, 'upper') # B is Kmm^{-1} Kmn

    #construct the mean and variance of q(f*)
    fmean = tf.matmul(tf.transpose(B), F)
    fvar = Kdiag[:,None] - tf.reduce_sum(tf.square(A), 0)[:,None]

    return fmean, fvar


def gaussian_gp_predict(Xnew, X, kern, q_mu, q_sqrt):
    """
    Given an (approximate) posterior (via q_mu, q_sqrt) to the GP at the points
    X, produce the mean and variance of the GP at the points Xnew.

    We assume K independent GPs, represented by the columns of q_mu (and the
    last ax of q_sqrt).  q_mu and q_sqrt are variational posteriors for f, So
        q(f[:,i]) = N (q_mu[:,i],  diag(q_sqrt[:,i]**2))
    or
        q(f[:,i]) = N (q_mu,  W W^T)
    where W is the lower triangle of q_sqrt[:,:,i]. 

    This function computes the Gaussian integral
        q(f*) = \int p(f*|f)q(f) df.

    Xnew is a data matrix, size N x D
    X are inducing points, size M x D
    q_mu are variational means, size M x K
    q_sqrt are variational standard-deviations or Cholesky matrices,, size M x K or M x M x K

    See also:
        gp_predict -- where there is no uncertainty in F (TODO)
        gaussian_gp_predict_whitened -- the same, but with whitening (centering) the f variables

    """
 
    #compute kernel stuff
    num_data = X.shape[0]
    Kdiag = kern.Kdiag(Xnew)
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + eye(num_data)*1e-4
    Lm = tf.cholesky(Kmm)

    #this is O(N M^2)
    A = tf.user_ops.triangular_solve(Lm, Kmn, 'lower')
    B = tf.user_ops.triangular_solve(tf.transpose(Lm), A, 'upper') # B is Kmm^{-1} Kmn

    #construct the mean and variance of q(f*)
    fmean = tf.matmul(tf.transpose(B), q_mu)
    fvar = Kdiag[:,None] - tf.reduce_sum(tf.square(A), 0)[:,None]
    if q_sqrt.get_shape().ndims==2:
        #we hae a diagonal form for q(f)
        fvar += tf.reduce_sum(tf.square(tf.transpose(B)[:,:,None] * q_sqrt[None,:,:]),1)
    elif q_sqrt.get_shape().ndims==3:
        # we have the cholesky form for q(v)
        def f(w):
            W = tf.triu(w)
            WB = tf.matmul(W, B)
            return tf.reduce_sum(tf.square(WB), 0)
        projected_var, _ = theano.scan(f, q_sqrt.swapaxes(0,2))
        fvar += projected_var.T

    return fmean, fvar

def gp_predict(Xnew, X, kern, F):
    """
    Given F, representing the GP at the points X, produce the mean and variance
    of the GP at the points Xnew.

    We assume K independent GPs, represented by the columns of F 
    . 
    This function computes the Gaussian conditional
        p(F* | F) 

    Xnew is a data matrix, size N x D
    X are inducing points, size M x D
    F are function values , size M x K

    See also:
        gp_predict_whitened -- where F is rotated into V (F = LV)
        gaussian_gp_predict -- similar, but with uncertainty in F

    """
 
    #compute kernel stuff
    num_data = X.shape[0]
    Kdiag = kern.Kdiag(Xnew)
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + eye(num_data)*1e-4
    Lm = tf.cholesky(Kmm)

    #this is O(N M^2)
    A = tf.user_ops.triangular_solve(Lm, Kmn, 'lower')
    B = tf.user_ops.triangular_solve(tf.transpose(Lm), A, 'upper') # B is Kmm^{-1} Kmn

    #construct the mean and variance of q(f*)
    fmean = tf.matmul(B.T, F)
    fvar = Kdiag[:,None] - tf.reduce_sum(tf.square(A), 0)[:,None]

    return fmean, fvar


def gaussian_gp_predict_whitened(Xnew, X, kern, q_mu, q_sqrt):
    """
    Given an (approximate) posterior (via q_mu, q_sqrt) to the GP at the points
    X, produce the mean and variance of the GP at the points Xnew.
    Additionally, the GP has been centered (whitened) so that 
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).

    We assume K independent GPs, represented by the columns of q_mu (and the
    last ax of q_sqrt).  q_mu and q_sqrt are variational posteriors for v, So
        q(v[:,i]) = N( q_mu[:,i], diag(q_sqrt[:,i]**2)
        q(f[:,i]) = N (L q_mu[:,i],  L diag(q_sqrt**2) L^T)
    or
        q(f[:,i]) = N (L q_mu,  L [W W^T] L^T)
    where W is the lower triangle of q_sqrt[:,:,i]. 

    This function computes the Gaussian integral
        q(f*) = \int p(f*|(f=Lv))q(v) df.

    Xnew is a data matrix, size N x D
    X are data points, size M x D
    q_mu are variational means, size M x K
    q_sqrt are variational standard-deviations or Cholesky matrices,, size M x K or M x M x K

    See also:
        gp_predict_whitened -- where there is no uncertainty in V
        gaussian_gp_predict -- same without the whitening

    """
 
    #compute kernel stuff
    num_data = X.shape[0]
    Kdiag = kern.Kdiag(Xnew)
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + eye(num_data)*1e-4
    Lm = tf.cholesky(Kmm)

    #this is O(N M^2)
    A = tf.user_ops.triangular_solve(Lm, Kmn, 'lower')

    #construct the mean and variance of q(f)
    fmean = tf.matmul(tf.transpose(A), q_mu)
    if q_sqrt.get_shape().ndims==2:
        #we hae a diagonal form for q(v)
        q_var = np.square(q_sqrt)
        #fvar = Kdiag[:,None] + tf.reduce_sum((tf.square(tf.transpose(A)))[:,:,None] * (q_var[None, :,:] - 1),1)
        fvar = tf.reshape(Kdiag, (-1,1)) + tf.reduce_sum(tf.reshape(tf.square(tf.transpose(A)), )[:,:,None] * (q_var[None, :,:] - 1),1)
    elif q_sqrt.get_shape().ndims ==3:
        # we have the cholesky form for q(v)
        fvar = Kdiag[:,None] - tf.reduce_sum(np.square(A), 0)[:,None]
        def f(w):
            R = tf.triu(w)
            RA = tf.matmul(R, A)
            return tf.square(RA).sum(0)
        projected_var, _ = theano.scan(f, q_sqrt.swapaxes(0,2))
        fvar += projected_var.T

    return fmean, fvar


def gp_predict_whitened(Xnew, X, kern, V):
    """
    Given a whitened representation of the GP at the points X (V), produce the
    mean and variance of the GP at the points Xnew (F*).

    The GP has been centered (whitened) so that 

        p(v) = N( 0, I)
        f = L v

    and so

        p(f) = N(0, LL^T) = N(0, K).

    We assume K independent GPs, represented by the columns of V. The GP consitional is:
    
        p(F*[:,i] | V[:,i]) = N (K_{*f} L^{-T} V[:,i],  K_{**} - K_{*f}L^{-1} L^{-T} K_{f*})

    Xnew is a data matrix, size N* x D
    X is a data matrix, size N x D
    V is a matrix containing whitened GP values, size N x K

    See also:
        gaussian_gp_predict_whitened -- where there is no uncertainty in V
        gp_predict -- without the whitening (TODO)
    """
    Kd = kern.Kdiag(Xnew)
    Kx = kern.K(X, Xnew)
    K = kern.K(X)
    L = tf.cholesky(K)
    A = tf.user_ops.triangular_solve(L, Kx, 'lower')
    fmean = tf.matmul(tf.transpose(A), V)
    fvar = Kd - tf.reduce_sum(tf.square(A), 0)
    return fmean, tf.expand_dims(fvar, 1) * tf.ones_like(V[0,:])



