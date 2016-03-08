import tensorflow as tf
import numpy as np
from .model import GPModel
from .param import Param
from .densities import multivariate_normal
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye

class SGPR(GPModel):
    def __init__(self, X, Y, kern, Z, mean_function=Zero()):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x multivariate_norma is an appropriate GPflow object

        Z is a matrix of inducing points, size M x D

        kern, mean_function are appropriate GPflow objects

        """
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.Z = Param(Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Constuct a tensorflow function to compute the bound on the marginal likelihood

        """

        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.shape(self.Y)[0]
        output_dim = tf.shape(self.Y)[1]

        err =  self.Y - self.mean_function(self.X)
        Kdiag = self.kern.Kdiag(self.X)
        Knm = self.kern.K(self.X, self.Z)
        Kmm = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        L = tf.cholesky(Kmm)
        beta = 1./self.likelihood.variance

        # Compute A
        tmp = tf.user_ops.triangular_solve(L, tf.transpose(Knm), 'lower')*tf.sqrt(beta)
        A_ = tf.matmul(tmp, tf.transpose(tmp))
        A = A_ + eye(num_inducing)
        LA = tf.cholesky(A)

        tmp = tf.user_ops.triangular_solve(L, tf.matmul(tf.transpose(Knm), err), 'lower')
        b = tf.user_ops.triangular_solve(LA, tmp, 'lower') * beta

        #compute log marginal bound
        bound = -0.5*tf.cast(num_data*output_dim, tf.float64)*np.log(2*np.pi)
        bound += -tf.cast(output_dim, tf.float64)*tf.reduce_sum(tf.log(tf.user_ops.get_diag(LA)))
        bound += 0.5*tf.cast(num_data*output_dim, tf.float64)*tf.log(beta)
        bound += -0.5*beta*tf.reduce_sum(tf.square(err))
        bound += 0.5*tf.reduce_sum(tf.square(b))
        bound += -0.5*(tf.reduce_sum(Kdiag)*beta - tf.reduce_sum(tf.user_ops.get_diag(A_)))

        return bound

    def build_predict(self, Xnew):
        raise NotImplementedError
