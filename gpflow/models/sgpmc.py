# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..base import Parameter
from ..conditionals import conditional
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction
from ..utilities import to_default_float
from .model import GPModel, InputData, MeanAndVariance, RegressionData
from .training_mixins import InternalDataTrainingLossMixin
from .util import data_input_to_tensor, inducingpoint_wrapper


class SGPMC(GPModel, InternalDataTrainingLossMixin):
    r"""
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

       \begin{align}
       \mathbf v & \sim N(0, \mathbf I) \\
       \mathbf u &= \mathbf L\mathbf v
       \end{align}

    with

    .. math::
        \mathbf L \mathbf L^\top = \mathbf K


    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        inducing_variable: Optional[InducingPoints] = None,
    ):
        """
        data is a tuple of X, Y with X, a data matrix, size [N, D] and Y, a data matrix, size [N, R]
        Z is a data matrix, of inducing inputs, size [M, D]
        kernel, likelihood, mean_function are appropriate GPflow objects
        """
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)
        self.num_data = data[0].shape[0]
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)
        self.V = Parameter(np.zeros((self.inducing_variable.num_inducing, self.num_latent_gps)))
        self.V.prior = tfp.distributions.Normal(
            loc=to_default_float(0.0), scale=to_default_float(1.0)
        )

    def log_posterior_density(self) -> tf.Tensor:
        return self.log_likelihood_lower_bound() + self.log_prior_density()

    def _training_loss(self) -> tf.Tensor:
        return -self.log_posterior_density()

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_likelihood_lower_bound()

    def log_likelihood_lower_bound(self) -> tf.Tensor:
        """
        This function computes the optimal density for v, q*(v), up to a constant
        """
        # get the (marginals of) q(f): exactly predicting!
        X_data, Y_data = self.data
        fmean, fvar = self.predict_f(X_data, full_cov=False)
        return tf.reduce_sum(self.likelihood.variational_expectations(fmean, fvar, Y_data))

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Xnew is a data matrix of the points at which we want to predict

        This method computes

            p(F* | (U=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at Z,

        """
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            self.V,
            full_cov=full_cov,
            q_sqrt=None,
            white=True,
            full_output_cov=full_output_cov,
        )
        return mu + self.mean_function(Xnew), var
