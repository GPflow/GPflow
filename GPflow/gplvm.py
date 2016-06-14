import tensorflow as tf
import numpy as np
from .model import GPModel
from .param import Param
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye
from . import kernel_expectations as ke
from . import transforms


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
        self.X_var = Param(X_var, transforms.positive)
        self.Z = Param(Z)
        self.num_data = X_mean.shape[0]
        self.num_latent = Z.shape[1]
        self.output_dim = Y.shape[1]

        assert X_mean.shape[0] == Y.shape[0], 'X mean and Y must be same size.'
        assert X_var.shape[0] == Y.shape[0], 'X var and Y must be same size.'

    def build_likelihood(self):
        """
        Constuct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.shape(self.Y)[0]
        output_dim = tf.shape(self.Y)[1]

        psi0, psi1, psi2 = ke.build_psi_stats(self.Z, self.kern, self.X_mean, self.X_var)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        ND = tf.cast(num_data*output_dim, tf.float64)
        D = tf.cast(output_dim, tf.float64)
        KL = -0.5*tf.reduce_sum(tf.log(self.X_var)) - 0.5 * ND +\
            0.5 * tf.reduce_sum(tf.square(self.X_mean) + self.X_var)

        # compute log marginal bound
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.diag_part(AAT)))
        bound -= KL

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        """
        raise NotImplementedError
