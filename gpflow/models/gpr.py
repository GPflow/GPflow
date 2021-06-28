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

from typing import Optional, Tuple

import tensorflow as tf

import gpflow

from .. import posteriors
from ..kernels import Kernel
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction
from ..posteriors import GPRPosterior
from ..utilities.model_utils import add_noise_cov
from .model import GPModel, InputData, MeanAndVariance, RegressionData
from .training_mixins import InternalDataTrainingLossMixin
from .util import data_input_to_tensor


class GPR_deprecated(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is given by

    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})

    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form

    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{K} + \sigma_n^2 \mathbf{I})
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X_data, Y_data = self.data
        K = self.kernel(X_data)
        ks = add_noise_cov(K, self.likelihood.variance)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X_data)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y_data, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)
        kmm_plus_s = add_noise_cov(kmm, self.likelihood.variance)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var


class GPR_with_posterior(GPR_deprecated):
    """
    This is an implementation of GPR that provides a posterior() method that
    enables caching for faster subsequent predictions.
    """

    def posterior(self, precompute_cache=posteriors.PrecomputeCacheType.TENSOR):
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        x_data, y_data = self.data

        return GPRPosterior(
            kernel=self.kernel,
            X_data=x_data,
            Y_data=y_data,
            likelihood_variance=self.likelihood.variance,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        For backwards compatibility, GPR's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.

        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


class GPR(GPR_with_posterior):
    # subclassed to ensure __class__ == "GPR"
    pass
