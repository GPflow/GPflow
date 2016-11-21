# -*- coding: utf-8 -*-

from .tf_wraps import eye
from .gpr import GPR
from .param import Param
from .transforms import Log1pe
from ._settings import settings

import numpy as np
import tensorflow as tf

class TP(GPR):
    def __init__(self, samples, values, **kwargs):
        super(TP, self).__init__(samples, values, **kwargs)

        self.likelihood.variance = 0.0
        self.likelihood.variance.fixed = True

        self._nu = Param(5.0, Log1pe(2.0))

    def build_likelihood(self):
        K = self.kern.K(self.X)
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        #
        d = self.Y - m
        alpha = tf.matrix_triangular_solve(L, d, lower=True)
        num_dims = tf.cast(tf.shape(self.Y)[0], settings.dtypes.float_type)
        ln_det_K = 2 * tf.reduce_sum(tf.diag_part(L))

        ret = -0.5 * num_dims * tf.log((self._nu - 2) * np.pi)
        ret += -0.5 * ln_det_K
        ret += tf.lgamma(0.5 * (self._nu + num_dims))
        ret += -tf.lgamma(0.5 * self._nu)
        ret += -0.5 * (self._nu + num_dims) * tf.log(1 + tf.reduce_sum(tf.square(alpha) / (self._nu - 2)))
        return ret

    def optimize(self, **kwargs):
        assert (self.likelihood.fixed)

        super(TP, self).optimize(**kwargs)

    def build_predict(self, Xnew, full_cov=False):
        pred_f_mean, pred_f_var = super(TP, self).build_predict(Xnew)

        K = self.kern.K(self.X)
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        d = self.Y - m
        alpha = tf.matrix_triangular_solve(L, d, lower=True)

        beta1 = tf.reduce_sum(tf.square(alpha))
        num_dims = tf.cast(tf.shape(self.Y)[0], settings.dtypes.float_type)

        pred_f_var = pred_f_var * (self._nu + beta1 - 2) / (self._nu + num_dims - 2)

        return pred_f_mean, pred_f_var
