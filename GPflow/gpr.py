import tensorflow as tf
from .model import GPModel
from .densities import multivariate_normal
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye


class GPR(GPModel):
    def __init__(self, X, Y, kern, mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects

        This is a vanilla implementation of GP regression with a Gaussian
        likelihood.  Multiple columns of Y are treated independently.
        """
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        
    def update_data(self, X, Y):
        """
        Update the data X and Y.
        If the shape changes, it will recompile but would work.
        """
        super(GPR, self).update_data(X, Y)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y, V | theta).

        """
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + eye(self.num_data) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar
