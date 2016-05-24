import tensorflow as tf
import numpy as np
from .model import GPModel
from .param import Param
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye
from .transforms import positive
from kernel_expectations import build_psi_stats


class BGPLVM(GPModel):

    def __init__(self, Xmean, Xvar, Y, kern, Z, mean_function=Zero()):
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, Xmean, Y, kern, likelihood, mean_function)
        self.Z = Param(Z)
        self.Xmean = Param(Xmean)
        self.Xvar = Param(Xvar, positive)

        self.num_data = Xmean.shape[0]
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.shape(self.Y)[0]
        output_dim = tf.shape(self.Y)[1]

        psi0, psi1, psi2 = build_psi_stats(self.Z, self.kern, self.Xmean, self.Xvar)
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
        KL = -0.5*tf.reduce_sum(tf.log(self.Xvar)) - 0.5 * ND +\
            0.5 * tf.reduce_sum(tf.square(self.Xmean) + self.Xvar)

        # compute log marginal bound
        bound = -0.5*ND*tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5*tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5*tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.diag_part(AAT)))
        bound -= KL

        return bound

    def build_predict(self, Xnew, full_cov=False):
        raise NotImplementedError
