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
        D = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True)*tf.sqrt(1./self.likelihood.variance)

        #TODO 

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        """
        raise NotImplementedError

