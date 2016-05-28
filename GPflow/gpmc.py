import numpy as np
import tensorflow as tf
from .model import GPModel
from .param import Param
from .conditionals import conditional
from .priors import Gaussian
from .mean_functions import Zero


class GPMC(GPModel):
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=Zero(), num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            f = Lv + m(x)

        with

            L L^T = K

        """
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.V = Param(np.zeros((self.num_data, self.num_latent)))
        self.V.prior = Gaussian(0., 1.)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y, V | theta).

        """
        K = self.kern.K(self.X)
        L = tf.cholesky(K)
        F = tf.matmul(L, self.V) + self.mean_function(self.X)

        return tf.reduce_sum(self.likelihood.logp(F, self.Y))

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern, self.V,
                              num_columns=self.num_latent, full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu + self.mean_function(Xnew), var
