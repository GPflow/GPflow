# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 2016

@author: keisukefujii
"""
import tensorflow as tf
from ..model import GPModel
from .. import gpr
from .model import Coregionalized_GPModel
from ..mean_functions import Zero
from . import likelihoods as coregionalized_likelihoods
from . import mean_functions as coregionalized_mean_functions
from ..likelihoods import Gaussian
from ..densities import multivariate_normal
from .labeled_data import LabeledData

class GPR(Coregionalized_GPModel, gpr.GPR):
    """
    Coregionalized GPR.
    
    This method inheritates from Coregionalized_GPModel and gpr.GPR.
    
    Coregionalized_GPModel provides some methods relating to the AutoFlow
    wrapping.
    """
    def __init__(self, X, Y, label, kern, mean_function=None, num_labels=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        label is a label for the data, size N
        kern should be one of coregionalized_kernels.
        """
        X = LabeledData((X, label), on_shape_change='pass', num_labels=num_labels)
        Y = LabeledData((Y, label), on_shape_change='pass', num_labels=num_labels)
        
        # Gaussian likelihoods for every labels
        likelihood = coregionalized_likelihoods.Likelihood(\
                            [Gaussian() for i in range(X.num_labels)])
        
        # If mean_function is None, Zero likelihoods are assumed.
        if mean_function is None:
            mean_function = coregionalized_mean_functions.MeanFunction(\
                                                     [Zero()]*X.num_labels)
        
        self.num_latent = Y.shape[1]
        
        # initialize GPModel rather than gpr
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)


    #TODO This method can be omitted by overriding operator in LabeledData.
    # and if tf.rank(self.Y), tf.shape(self.Y) can be hacked in some way...
    def build_likelihood(self):
        """
        This method is again defined to replace
        self.Y -> self.Y.data
        in multivariate_normal(self.Y, m, L)
        """
        K = self.kern.K(self.X) + self.get_variance()
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y.data, m, L)
        
    # TODO This method may be omitted by overriding operator in LabeledData
    # and if tf.shape(self.Y) can be hacked in some way...
    def build_predict(self, Xnew, full_cov=False):
        """
        This method is again defined to replace
        self.Y -> self.Y.data
        in 
        V = tf.matrix_triangular_solve(L, self.Y.data - self.mean_function(self.X))
        shape = tf.pack([1, 1, tf.shape(self.Y.data)[1]])
        and
        fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y.data)[1]])
        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + self.get_variance()
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y.data - self.mean_function(self.X))
        fmean = tf.matmul(tf.transpose(A), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(tf.transpose(A), A)
            shape = tf.pack([1, 1, tf.shape(self.Y.data)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y.data)[1]])
        return fmean, fvar
        
        
    def get_variance(self):
        """
        Overload get_variance method so that the variance-vector from multiple
        likelihoods is appropriately gathered.
        """
        var = []
        for y, lik in zip(self.Y.split(self.Y.data), self.likelihood):
            var.append(tf.squeeze(tf.ones(tf.shape(y), dtype=tf.float64)*lik.variance))
        return tf.diag(self.Y.restore(var))
        