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
from gpflow.features import InducingPoints
from ..conditionals import conditional
from ..models.model import GPModelOLD, MeanAndVariance


class SGPMC(GPModelOLD):
    """
    This is the Sparse Variational GP using MCMC (SGPMC). The key reference is

    ::

      @inproceedings{hensman2015mcmc,
        title={MCMC for Variatinoally Sparse Gaussian Processes},
        author={Hensman, James and Matthews, Alexander G. de G.
                and Filippone, Maurizio and Ghahramani, Zoubin},
        booktitle={Proceedings of NIPS},
        year={2015}
      }

    The latent function values are represented by centered
    (whitened) variables, so

    .. math::
       :nowrap:

       \\begin{align}
       \\mathbf v & \\sim N(0, \\mathbf I) \\\\
       \\mathbf u &= \\mathbf L\\mathbf v
       \\end{align}

    with

    .. math::
        \\mathbf L \\mathbf L^\\top = \\mathbf K


    """

    def __init__(self,
                 X,
                 Y,
                 kernel,
                 likelihood,
                 mean_function=None,
                 num_latent=None,
                 features=None,
                 **kwargs):
        """
        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        Z is a data matrix, of inducing inputs, size [M, D]
        kernel, likelihood, mean_function are appropriate GPflow objects
        """
        GPModelOLD.__init__(self,
                            X,
                            Y,
                            kernel,
                            likelihood,
                            mean_function,
                            num_latent=num_latent,
                            **kwargs)
        self.num_data = X.shape[0]
        self.feature = InducingPoints(features)
        self.V = Parameter(np.zeros((len(self.feature), self.num_latent)))
        self.V.prior = tfp.distributions.Normal(loc=0., scale=1.)

    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        """
        This function computes the optimal density for v, q*(v), up to a constant
        """
        # get the (marginals of) q(f): exactly predicting!
        fmean, fvar = self.predict_f(self.X, full_cov=False)
        return tf.reduce_sum(
            self.likelihood.variational_expectations(fmean, fvar, self.Y))

    def predict_f(self, X: tf.Tensor, full_cov=False,
                  full_output_cov=False) -> MeanAndVariance:
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (U=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at Z,

        """
        mu, var = conditional(X,
                              self.feature,
                              self.kernel,
                              self.V,
                              full_cov=full_cov,
                              q_sqrt=None,
                              white=True,
                              full_output_cov=full_output_cov)
        return mu + self.mean_function(X), var
