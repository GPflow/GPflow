# -*- coding: utf-8 -*-
import tensorflow as tf
from .coregionalized_param import LabeledAutoFlow
from .labeled_data import LabeledData
from .data_holders import ScalarData
from .. import tf_hacks

class Coregionalized_GPModel(object):
    """
    A base class for coregionalized Gaussian process models.
    
    This class is added to modify some methods in model.GPModel.
    
    The inheriting class, e.g. coregionalized_gpr.GPR, should inherite from 
    this class as well as model.GPModel
    
    >>> class coregionalized_gpr.GPR(Coregionalized_GPModel, GPModel):
    >>>     ...
    
    
    AutoFlow methods are overloaded to handled the labeled data.

    The argument of these method is a tuple of data and index, but this tuple 
    is converted to LabeledData by LabeledAutoFlow wrapping.

    """

    def __init__(self):
        """
        This object has no instances (except for those added by AutoFlow).
        """
        pass
    
    @property
    def num_labels(self):
        return self.X.num_labels
    
    @LabeledAutoFlow(LabeledData)
    def predict_f(self, Xnew_index_tuple):
        return self.build_predict(Xnew_index_tuple)

    @LabeledAutoFlow(LabeledData)
    def predict_f_full_cov(self, Xnew_index_tuple):
        return self.build_predict(Xnew_index_tuple, full_cov=True)

    @LabeledAutoFlow(LabeledData, ScalarData)
    def predict_f_samples(self, Xnew_index_tuple, num_samples):
        mu, var = self.build_predict(Xnew_index_tuple, full_cov=True)
        jitter = tf_hacks.eye(tf.shape(mu)[0]) * 1e-6
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.pack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=tf.float64)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.pack(samples))

    @LabeledAutoFlow(LabeledData)
    def predict_y(self, Xnew_index_tuple):
        pred_f_mean, pred_f_var = self.build_predict(Xnew_index_tuple)
        # Labeled data is also passed to predict_mean_and_var to distinguish
        # which likelihood to be used for each data.
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var, Xnew_index_tuple)

    @LabeledAutoFlow(LabeledData, LabeledData)
    def predict_density(self, Xnew_index_tuple, Ynew_index_tuple):
        pred_f_mean, pred_f_var = self.build_predict(Xnew_index_tuple)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew_index_tuple)
