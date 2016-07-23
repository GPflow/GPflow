# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from .. import gpr
from ..model import GPModel
from ..densities import multivariate_normal
from ..mean_functions import Zero
from . import coregionalized_likelihoods
from ..likelihoods import Gaussian
from .labeled_data import LabeledData

class GPR(gpr.GPR):
    """
    Coregionalized GPR.
    """
    def __init__(self, X, Y, label, kern, mean_function=Zero(), num_labels=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        label is a label for the data, size N
        kern should be one of coregionalized_kernels.
        """
        X = LabeledData(X, label, on_shape_change='pass', num_labels=num_labels)
        Y = LabeledData(Y, label, on_shape_change='pass', num_labels=num_labels)
        
        # Gaussian likelihoods for every labels
        likelihood = coregionalized_likelihoods.Likelihood(\
                            [Gaussian() for i in range(len(X.num_labels))])
        
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_latent = Y.data.shape[1]
        
    
    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y, V | theta).
        """
        K = self.kern.K(self.X) + self.get_variation()
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y.data, m, L)


    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + eye(tf.shape(self.X)[0]) * self.likelihood.variance
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

    def get_variance(self):
        """
        Construct variance tensor for this likelihood.
        """
        var = []
        for y, lik in zip(self.Y.split(self.Y.data), self.likelihood):
            var.append(tf.ones(tf.shape(y))*lik.variation)
        return tf.diag(self.Y.restore(var))
        