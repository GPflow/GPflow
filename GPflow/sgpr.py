import tensorflow as tf
import numpy as np
from .model import GPModel
from .param import Param
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye


class SGPR(GPModel):
    def __init__(self, X, Y, kern, Z, mean_function=Zero()):
        """
        This Sparse Variational GP regression. The key reference is

        @inproceedings{titsias2009variational,
          title={Variational learning of inducing variables in
                 sparse Gaussian processes},
          author={Titsias, Michalis K},
          booktitle={International Conference on
                     Artificial Intelligence and Statistics},
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
        Constuct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """

        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.cast(tf.shape(self.Y)[0], tf.float64)
        output_dim = tf.cast(tf.shape(self.Y)[1], tf.float64)

        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        L = tf.cholesky(Kuu)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        AAT = tf.matmul(A, tf.transpose(A))
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5*tf.cast(num_data*output_dim, tf.float64)*np.log(2*np.pi)
        bound += -output_dim * tf.reduce_sum(tf.log(tf.user_ops.get_diag(LB)))
        bound -= 0.5 * num_data * output_dim * tf.log(self.likelihood.variance)
        bound += -0.5*tf.reduce_sum(tf.square(err))/self.likelihood.variance
        bound += 0.5*tf.reduce_sum(tf.square(c))
        bound += -0.5 * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.user_ops.get_diag(AAT))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        num_inducing = tf.shape(self.Z)[0]
        err = self.Y - self.mean_function(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        Kus = self.kern.K(self.Z, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        B = tf.matmul(A, tf.transpose(A)) + eye(num_inducing)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tf.transpose(tmp2), tmp2)\
                - tf.matmul(tf.transpose(tmp1), tmp1)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0)\
                - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.pack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var
