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
import warnings
from typing import Callable, Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf

from ..base import Module
from ..config import default_float, default_jitter
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction, Zero
from ..utilities import ops

InputData = tf.Tensor
OutputData = tf.Tensor
RegressionData = Tuple[InputData, OutputData]
Data = TypeVar("Data", RegressionData, InputData)
MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]


class BayesianModel(Module, metaclass=abc.ABCMeta):
    """ Bayesian model. """

    def log_prior_density(self) -> tf.Tensor:
        log_priors = [p.log_prior() for p in self.trainable_parameters]
        if log_priors:
            return tf.add_n(log_priors)
        else:
            return tf.convert_to_tensor(0.0, dtype=default_float())

    def log_posterior_density(self, data: Optional[Data] = None) -> tf.Tensor:
        """
        This may be the posterior with respect to the hyperparameters (e.g. for
        GPR) or the posterior with respect to the function (e.g. for GPMC and
        SGPMC). It assumes that maximum_likelihood_objective() is defined
        sensibly.
        """
        return self._call_maximum_likelihood_objective(data) + self.log_prior_density()

    def training_loss(self, data: Optional[Data] = None) -> tf.Tensor:
        """
        Minimization objective for TensorFlow optimizers (including
        gpflow.optimizers.Scipy). Includes the log prior density for maximum
        a-posteriori (MAP) estimation.
        """
        return -(self._call_maximum_likelihood_objective(data) + self.log_prior_density())

    def training_loss_closure(self, data: Optional[Data] = None) -> Callable[[], tf.Tensor]:
        def training_loss_closure():
            return self.training_loss(data)

        return training_loss_closure

    def _call_maximum_likelihood_objective(self, data: Optional[Data] = None) -> tf.Tensor:
        """
        Helper function so that a user's model will still work even if they
        implement maximum_likelihood_objective() without any arguments
        """
        if data is None:
            return self.maximum_likelihood_objective()
        else:
            return self.maximum_likelihood_objective(data)

    @abc.abstractmethod
    def maximum_likelihood_objective(self, data: Optional[Data] = None) -> tf.Tensor:
        """
        Objective for maximum likelihood estimation. Should be maximized. E.g.
        log-marginal likelihood (hyperparameter likelihood) for GPR, or lower
        bound to the log-marginal likelihood (ELBO) for sparse and variational
        GPs.
        """
        raise NotImplementedError


class GPModel(BayesianModel):
    r"""
    A stateless base class for Gaussian process models, that is, those of the
    form

    .. math::
       :nowrap:

       \begin{align}
           \theta        & \sim p(\theta) \\
           f             & \sim \mathcal{GP}(m(x), k(x, x'; \theta)) \\
           f_i           & = f(x_i) \\
           y_i \,|\, f_i & \sim p(y_i|f_i)
       \end{align}

    This class mostly adds functionality for predictions. To use it, inheriting
    classes must define a predict_f function, which computes the means and
    variances of the latent function.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_log_density.

    It is also possible to draw samples from the latent GPs using
    self.predict_f_samples.
    """

    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,
    ):
        super().__init__()
        self.num_latent_gps = num_latent_gps
        # TODO(@awav): Why is this here when MeanFunction does not have a __len__ method
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.kernel = kernel
        self.likelihood = likelihood

    @abc.abstractmethod
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError

    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: int = 1,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.
        """
        mu, var = self.predict_f(Xnew, full_cov=full_cov)  # [N, P], [P, N, N]
        num_latent_gps = var.shape[0]
        num_elems = tf.shape(var)[1]
        var_jitter = ops.add_to_diagonal(var, default_jitter())
        L = tf.linalg.cholesky(var_jitter)  # [P, N, N]
        V = tf.random.normal([num_latent_gps, num_elems, num_samples], dtype=mu.dtype)  # [P, N, S]
        LV = L @ V  # [P, N, S]
        mu_t = tf.linalg.adjoint(mu)  # [P, N]
        return tf.transpose(mu_t[..., np.newaxis] + LV)  # [S, N, P]

    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_log_density(
        self, data: RegressionData, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Compute the log density of the data at the new data points.
        """
        X, Y = data
        f_mean, f_var = self.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_density(f_mean, f_var, Y)
