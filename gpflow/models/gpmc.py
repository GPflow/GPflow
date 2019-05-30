# Copyright 2016 James Hensman, alexggmatthews
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

from gpflow.base import Parameter
from gpflow.utilities.defaults import default_float, default_jitter

from ..conditionals import conditional
from .model import GPModelOLD, MeanAndVariance


class GPMC(GPModelOLD):
    def __init__(self,
                 X,
                 Y,
                 kernel,
                 likelihood,
                 mean_function=None,
                 num_latent=None,
                 **kwargs):
        """
        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        kernel, likelihood, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            f = Lv + m(x)

        with

            L L^T = K

        """
        GPModelOLD.__init__(self, X, Y, kernel, likelihood, mean_function,
                            num_latent, **kwargs)
        self.num_data = X.shape[0]
        self.V = Parameter(np.zeros((self.num_data, self.num_latent)))
        self.V.prior = tfp.distributions.Normal(loc=0., scale=1.)

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        """
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y, V | theta).

        """
        K = self.kernel(self.X)
        L = tf.linalg.cholesky(
            K + tf.eye(tf.shape(self.X)[0], dtype=default_float()) *
            default_jitter())
        F = tf.linalg.matmul(L, self.V) + self.mean_function(self.X)

        return tf.reduce_sum(self.likelihood.log_prob(F, self.Y))

    def predict_f(self, Xnew: tf.Tensor, full_cov=False,
                  full_output_cov=False) -> MeanAndVariance:
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        mu, var = conditional(Xnew,
                              self.X,
                              self.kernel,
                              self.V,
                              full_cov=full_cov,
                              q_sqrt=None,
                              white=True)
        return mu + self.mean_function(Xnew), var
