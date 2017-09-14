# Copyright 2016 James Hensman, Mark van der Wilk, Valentine Svensson, alexggmatthews, fujiisoup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf

from . import hmc

from .params import Parameterized
from .autoflow import AutoFlow
from .mean_functions import Zero
from .misc import TF_FLOAT_TYPE
from ._settings import settings


class Model(Parameterized):
    def __init__(self, name=None):
        """
        Name is a string describing this model.
        """
        super(Model, self).__init__(name=name)
        self._likelihood_tensor = None
        self._objective = None
        self._objective_gradient = None

    @property
    def objective(self):
        return self._objective

    @property
    def objective_gradient(self):
        return self._objective_gradient

    @AutoFlow()
    def compute_log_prior(self):
        """ Compute the log prior of the model (uses AutoFlow)"""
        return self.prior_tensor

    @AutoFlow()
    def compute_log_likelihood(self):
        """ Compute the log likelihood of the model (uses AutoFlow on ``self.build_likelihood()``)"""
        return self.likelihood_tensor

    #def sample(self, num_samples, Lmin=5, Lmax=20, epsilon=0.01, thin=1, burn=0,
    #           verbose=False, return_logprobs=False, RNG=np.random.RandomState(0)):
    #    """
    #    Use Hamiltonian Monte Carlo to draw samples from the model posterior.
    #    """
    #    if self._needs_recompile:
    #        self._compile()
    #    return hmc.sample_HMC(self._objective, num_samples,
    #                          Lmin=Lmin, Lmax=Lmax, epsilon=epsilon, thin=thin, burn=burn,
    #                          x0=self.get_free_state(), verbose=verbose,
    #                          return_logprobs=return_logprobs, RNG=RNG)

    def _build(self):
        super(Model, self)._build()

        self._likelihood_tensor = self._build_likelihood()
        func = tf.add(self.likelihood_tensor, self.prior_tensor)
        grad_func = tf.gradients(func, self.trainable_tensors)

        self._objective = tf.negative(func, name='objective')
        self._objective_gradient = tf.negative(grad_func, name='objective_gradient')

    def _build_likelihood(self, *args, **kwargs):
        raise NotImplementedError()


class GPModel(Model):
    """
    A base class for Gaussian process models, that is, those of the form

    .. math::
       :nowrap:

       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \\end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.

    For handling another data (Xnew, Ynew), set the new value to self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(self, X, Y, kern, likelihood, mean_function, name=None):
        super(GPModel, self).__init__(name=name)
        self.mean_function = mean_function or Zero()
        self.kern = kern
        self.likelihood = likelihood

        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        self.X, self.Y = X, Y

    def _build_predict(self, *args, **kwargs):
        raise NotImplementedError

    @AutoFlow((TF_FLOAT_TYPE, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self.build_predict(Xnew)

    @AutoFlow((TF_FLOAT_TYPE, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.build_predict(Xnew, full_cov=True)

    @AutoFlow((TF_FLOAT_TYPE, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=TF_FLOAT_TYPE) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=TF_FLOAT_TYPE)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @AutoFlow((TF_FLOAT_TYPE, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((TF_FLOAT_TYPE, [None, None]), (TF_FLOAT_TYPE, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)
