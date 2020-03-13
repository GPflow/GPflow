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
from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Sequence, Callable

import numpy as np
import tensorflow as tf

from .. import Parameter
from ..config import default_float, default_jitter
from ..likelihoods import Likelihood
from ..utilities import ops

Data = TypeVar("Data", Tuple[tf.Tensor, tf.Tensor], tf.Tensor)
DataPoint = tf.Tensor
MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]


class Prior(ABC):
    @property
    @abstractmethod
    def trainable_parameters(self) -> Sequence[Parameter]:
        pass

    def log_prior_density(self) -> tf.Tensor:
        return (
            tf.add_n([p.log_prior() for p in self.trainable_parameters])
            if self.trainable_parameters
            else tf.convert_to_tensor(0., dtype=default_float())
        )


class BayesianModel(ABC, Prior):
    def log_posterior_density(self, data: ...) -> tf.Tensor:
        return self.maximum_likelihood_objective(data) + self.log_prior_density()

    @abstractmethod
    def maximum_likelihood_objective(self, data: ...) -> tf.Tensor:
        pass

    def training_loss_closure(self, data: ...) -> Callable[[], tf.Tensor]:
        return lambda: - self.maximum_likelihood_objective(data) - self.log_prior_density()


class BayesianModelWithData(ABC, Prior):
    def log_posterior_density(self) -> tf.Tensor:
        return self.maximum_likelihood_objective() + self.log_prior_density()

    @abstractmethod
    def maximum_likelihood_objective(self) -> tf.Tensor:
        pass

    def training_loss_closure(self) -> Callable[[], tf.Tensor]:
        return lambda: - self.maximum_likelihood_objective() - self.log_prior_density()


class GPModel(ABC):
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
    @property
    @abstractmethod
    def likelihood(self) -> Likelihood:
        pass

    @abstractmethod
    def predict_f(
        self, predict_at: DataPoint, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError

    def predict_f_samples(
        self,
        predict_at: DataPoint,
        num_samples: int = 1,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.
        """
        mu, var = self.predict_f(predict_at, full_cov=full_cov)  # [N, P], [P, N, N]
        num_latent_gps = var.shape[0]
        num_elems = tf.shape(var)[1]
        var_jitter = ops.add_to_diagonal(var, default_jitter())
        L = tf.linalg.cholesky(var_jitter)  # [P, N, N]
        V = tf.random.normal([num_latent_gps, num_elems, num_samples], dtype=mu.dtype)  # [P, N, S]
        LV = L @ V  # [P, N, S]
        mu_t = tf.linalg.adjoint(mu)  # [P, N]
        return tf.transpose(mu_t[..., np.newaxis] + LV)  # [S, N, P]

    def predict_y(
        self, predict_at: DataPoint, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        f_mean, f_var = self.predict_f(
            predict_at, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_log_density(
        self, data: Data, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Compute the log density of the data at the new data points.
        """
        x, y = data
        f_mean, f_var = self.predict_f(x, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_density(f_mean, f_var, y)
