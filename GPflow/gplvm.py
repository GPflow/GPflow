import tensorflow as tf
import numpy as np
from .model import GPModel
from .param import Param
from .densities import multivariate_normal
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye

class BayesianGPLVM(GPModel):

    def __init__(self, X_mean, X_var, Y, kern, Z):
        """
        X_mean is a data matrix, size N x D
        X_var is a data matrix, size N x D (X_var > 0) 
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """
        GPModel.__init__(self, X_mean, Y, kern, likelihood=likelihoods.Gaussian(), mean_function=Zero())
        del self.X
        self.X_mean = Param(X_mean)
        self.X_var = Param(X_var)
        self.Z = Param(Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Constuct a tensorflow function to compute the bound on the marginal
        likelihood. 
        """

        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.shape(self.Y)[0]
        output_dim = tf.shape(self.Y)[1]

        err =  self.Y - self.mean_function(self.X)
        psi0, psi1, psi2 = build_psi_stats(self.Z, self.kern, self.X_mean, self.X_var)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        L = tf.cholesky(Kuu)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / tf.sqrt(self.likelihood.variance)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tf.matrix_triangular_solve(L, psi2, lower=True)), lower=True)/self.likelihood.variance
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / tf.sqrt(self.likelihood.variance)

        #compute log marginal bound
        bound = -0.5*tf.cast(num_data*output_dim, tf.float64)*np.log(2*np.pi*self.likelihood.variance)
        bound += -tf.cast(output_dim, tf.float64)*tf.reduce_sum(tf.log(tf.user_ops.get_diag(LB)))
        bound += -0.5*tf.reduce_sum(tf.square(self.Y))/self.likelihood.variance
        bound += 0.5*tf.reduce_sum(tf.square(c))
        bound += -0.5*tf.cast(tf.float64, self.output_dim)*(psi0/self.likelihood.variance - tf.reduce_sum(tf.user_ops.get_diag(AAT)))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        """
        raise NotImplementedError

