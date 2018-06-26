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

import abc

import numpy as np
import tensorflow as tf

from .. import settings
from ..core.compilable import Build
from ..params import Parameterized, DataHolder
from ..decors import autoflow
from ..mean_functions import Zero


class Model(Parameterized):
    def __init__(self, name=None):
        """
        Name is a string describing this model.
        """
        super(Model, self).__init__(name=name)
        self._objective = None
        self._likelihood_tensor = None

    @property
    def objective(self):
        return self._objective

    @property
    def likelihood_tensor(self):
        return self._likelihood_tensor

    @autoflow()
    def compute_log_prior(self):
        """Compute the log prior of the model."""
        return self.prior_tensor

    @autoflow()
    def compute_log_likelihood(self):
        """Compute the log likelihood of the model."""
        return self.likelihood_tensor

    def is_built(self, graph):
        is_built = super().is_built(graph)
        if is_built is not Build.YES:
            return is_built
        if self._likelihood_tensor is None:
            return Build.NO
        return Build.YES

    def build_objective(self):
        likelihood = self._build_likelihood()
        priors = []
        for param in self.parameters:
            unconstrained = param.unconstrained_tensor
            constrained = param._build_constrained(unconstrained)
            priors.append(param._build_prior(unconstrained, constrained))
        prior = self._build_prior(priors)
        return self._build_objective(likelihood, prior)

    def _clear(self):
        super(Model, self)._clear()
        self._likelihood_tensor = None
        self._objective = None

    def _build(self):
        super(Model, self)._build()
        likelihood = self._build_likelihood()
        prior = self.prior_tensor
        objective = self._build_objective(likelihood, prior)
        self._likelihood_tensor = likelihood
        self._objective = objective

    def sample_feed_dict(self, sample):
        tensor_feed_dict = {}
        for param in self.parameters:
            if not param.trainable: continue
            constrained_value = sample[param.pathname]
            unconstrained_value = param.transform.backward(constrained_value)
            tensor = param.unconstrained_tensor
            tensor_feed_dict[tensor] = unconstrained_value
        return tensor_feed_dict

    def _build_objective(self, likelihood_tensor, prior_tensor):
        func = tf.add(likelihood_tensor, prior_tensor, name='nonneg_objective')
        return tf.negative(func, name='objective')

    @abc.abstractmethod
    def _build_likelihood(self):
        raise NotImplementedError('')  # TODO(@awav): write error message


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

    def __init__(self, X, Y, kern, likelihood, mean_function,
                 num_latent=None, name=None):
        super(GPModel, self).__init__(name=name)
        self.num_latent = num_latent or Y.shape[1]
        self.mean_function = mean_function or Zero(output_dim=self.num_latent)
        self.kern = kern
        self.likelihood = likelihood

        if isinstance(X, np.ndarray):
            # X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            # Y is a data matrix, rows correspond to the rows in X,
            # columns are treated independently
            Y = DataHolder(Y)
        self.X, self.Y = X, Y

    @autoflow((settings.float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self._build_predict(Xnew)

    @autoflow((settings.float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self._build_predict(Xnew, full_cov=True)

    @autoflow((settings.float_type, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self._build_predict(Xnew, full_cov=True)  # N x P, # P x N x N
        jitter = tf.eye(tf.shape(mu)[0], dtype=settings.float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[i, :, :] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @autoflow((settings.float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @autoflow((settings.float_type, [None, None]), (settings.float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    @abc.abstractmethod
    def _build_predict(self, *args, **kwargs):
        raise NotImplementedError('') # TODO(@awav): write error message
