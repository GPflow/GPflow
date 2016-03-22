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
        This Sparse Variational GP regression. The key reference is

        @inproceedings{titsias2009variational,
          title={Variational learning of inducing variables in sparse Gaussian processes},
          author={Titsias, Michalis K},
          booktitle={International Conference on Artificial Intelligence and Statistics},
          pages={567--574},
          year={2009}
        }


        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

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
        tmp = tf.matrix_triangular_solve(L, tf.transpose(Knm), lower=True)*tf.sqrt(1./self.likelihood.variance)
        A_ = tf.matmul(tmp, tf.transpose(tmp))
        A = A_ + eye(num_inducing)
        LA = tf.cholesky(A)

        tmp = tf.matrix_triangular_solve(L, tf.matmul(tf.transpose(Knm), err), lower=True)
        b = tf.matrix_triangular_solve(LA, tmp, lower=True) * beta

        #compute log marginal bound
        bound = -0.5*tf.cast(num_data*output_dim, tf.float64)*np.log(2*np.pi)
        bound += -tf.cast(output_dim, tf.float64)*tf.reduce_sum(tf.log(tf.user_ops.get_diag(LA)))
        bound += 0.5*tf.cast(num_data*output_dim, tf.float64)*tf.log(beta)
        bound += -0.5*beta*tf.reduce_sum(tf.square(err))
        bound += 0.5*tf.reduce_sum(tf.square(b))
        bound += -0.5*(tf.reduce_sum(Kdiag)*beta - tf.reduce_sum(tf.user_ops.get_diag(A_)))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        err =  self.Y - self.mean_function(self.X)
        beta = 1./self.likelihood.variance
        Knm = self.kern.K(self.X, self.Z)
        num_inducing = tf.shape(self.Z)[0]
        Kmm = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        Kus = self.kern.K(self.Z, Xnew)
        L = tf.cholesky(Kmm)
        B = tf.matrix_triangular_solve(L, tf.transpose(Knm), lower=True)*tf.sqrt(beta)
        A = tf.matmul(B, tf.transpose(B)) + eye(num_inducing)
        LA = tf.cholesky(A)
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LA, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), tf.matrix_triangular_solve(LA, tf.matmul(B, err*tf.sqrt(beta)), lower=True))
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tf.transpose(tmp2), tmp2) - tf.matmul(tf.transpose(tmp1), tmp1)
            var = tf.tile(tf.expand_dims(var, 2), tf.pack([1,1, tf.shape(self.Y)[1]]))
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) - tf.reduce_sum(tf.square(tmp1), 0)
            var = tf.tile(tf.expand_dims(var, 1), tf.pack([1, tf.shape(self.Y)[1]]))
        return mean + self.mean_function(Xnew), var

