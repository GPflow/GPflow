# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..base import Module, Parameter
from ..config import default_float
from ..quadrature import hermgauss
from ..utilities import to_default_float, to_default_int
from .base import Likelihood, MonteCarloLikelihood


class Softmax(MonteCarloLikelihood):
    """
    The soft-max multi-class likelihood.  It can only provide a stochastic
    Monte-Carlo estimate of the variational expectations term, but this
    added variance tends to be small compared to that due to mini-batching
    (when using the SVGP model).
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(latent_dim=num_classes, observation_dim=None, **kwargs)
        self.num_classes = self.latent_dim

    def _log_prob(self, F, Y):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=Y[:, 0])

    def _conditional_mean(self, F):
        return tf.nn.softmax(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - p ** 2


class RobustMax(Module):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-epsilon)  i == argmax(f)
          epsilon/(k-1)  otherwise

    where k is the number of classes.
    """

    def __init__(self, num_classes, epsilon=1e-3, **kwargs):
        """
        `epsilon` represents the fraction of 'errors' in the labels of the
        dataset. This may be a hard parameter to optimize, so by default
        it is set un-trainable, at a small value.
        """
        super().__init__(**kwargs)
        transform = tfp.bijectors.Sigmoid()
        prior = tfp.distributions.Beta(to_default_float(0.2), to_default_float(5.0))
        self.epsilon = Parameter(epsilon, transform=transform, prior=prior, trainable=False)
        self.num_classes = num_classes
        self._squash = 1e-6

    def __call__(self, F):
        i = tf.argmax(F, 1)
        return tf.one_hot(
            i, self.num_classes, tf.squeeze(1.0 - self.epsilon), tf.squeeze(self.eps_k1)
        )

    @property
    def eps_k1(self):
        return self.epsilon / (self.num_classes - 1.0)

    def safe_sqrt(self, val):
        return tf.sqrt(tf.clip_by_value(val, 1e-10, np.inf))

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = to_default_int(Y)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(
            tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1.0, 0.0), dtype=mu.dtype
        )
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            self.safe_sqrt(2.0 * var_selected), (-1, 1)
        )

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            self.safe_sqrt(var), 2
        )
        cdfs = 0.5 * (1.0 + tf.math.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2 * self._squash) + self._squash

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(
            tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0.0, 1.0), dtype=mu.dtype
        )
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.reduce_prod(cdfs, axis=[1]) @ tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1))


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None, **kwargs):
        """
        A likelihood for multi-way classification.  Currently the only valid
        choice of inverse-link function (invlink) is an instance of RobustMax.

        For most problems, the stochastic `Softmax` likelihood may be more
        appropriate (note that you then cannot use Scipy optimizer).
        """
        super().__init__(latent_dim=num_classes, observation_dim=None, **kwargs)
        self.num_classes = num_classes
        self.num_gauss_hermite_points = 20

        if invlink is None:
            invlink = RobustMax(self.num_classes)

        if not isinstance(invlink, RobustMax):
            raise NotImplementedError

        self.invlink = invlink

    def _log_prob(self, F, Y):
        hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
        yes = tf.ones(tf.shape(Y), dtype=default_float()) - self.invlink.epsilon
        no = tf.zeros(tf.shape(Y), dtype=default_float()) + self.invlink.eps_k1
        p = tf.where(hits, yes, no)
        return tf.reduce_sum(tf.math.log(p), axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        ve = p * tf.math.log(1.0 - self.invlink.epsilon) + (1.0 - p) * tf.math.log(
            self.invlink.eps_k1
        )
        return tf.reduce_sum(ve, axis=-1)

    def _predict_mean_and_var(self, Fmu, Fvar):
        possible_outputs = [
            tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64))
            for i in range(self.num_classes)
        ]
        ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
        ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
        return ps, ps - tf.square(ps)

    def _predict_log_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(tf.math.log(self._predict_non_logged_density(Fmu, Fvar, Y)), axis=-1)

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        den = p * (1.0 - self.invlink.epsilon) + (1.0 - p) * (self.invlink.eps_k1)
        return den

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)
