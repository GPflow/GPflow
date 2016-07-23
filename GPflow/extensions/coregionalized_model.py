# -*- coding: utf-8 -*-
import tensorflow as tf
from ..param import AutoFlow

class Coregionalized_GPModel(object):
    """
    A base class for coregionalized Gaussian process models.
    
    This class is added to modify some methods in model.GPModel.
    
    The inheriting class, e.g. coregionalized_gpr.GPR, should inherite from 
    this class as well as model.GPModel
    
    >>> class coregionalized_gpr.GPR(Coregionalized_GPModel, GPModel):
    >>>     ...
    
    """

    def __init__(self):
        """
        This class should not have any instances.
        """
        pass
    
    @LabeledAutoFlow({'data':(tf.float64, [None, None])}, {'label':(tf.int32, [None])})
    def predict_f(self, Xnew, label_new):
        # TODO 
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew labeled with label_new.
        
        By CoregionalizedAutoFlow wrapping, Xnew becomes LabeledData under 
        tf_mode.
        """
        return self.build_predict(Xnew)

    @AutoFlow((tf.float64, [None, None]), (tf.int32, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        # TODO Imprement
        raise NotImplementedError
        return self.build_predict(Xnew, full_cov=True)

    @AutoFlow((tf.float64, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        # TODO Imprement
        raise NotImplementedError

        mu, var = self.build_predict(Xnew, full_cov=True)
        jitter = tf_hacks.eye(tf.shape(mu)[0]) * 1e-6
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.pack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=tf.float64)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.pack(samples))

    @AutoFlow((tf.float64, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        # TODO Imprement
        raise NotImplementedError

        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((tf.float64, [None, None]), (tf.float64, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        # TODO Imprement
        raise NotImplementedError

        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)
