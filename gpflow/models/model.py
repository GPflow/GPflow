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
from typing import Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf

from ..base import Module
from ..conditionals.util import sample_mvn
from ..config import default_float, default_jitter
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction, Zero
from ..utilities import ops

Data = TypeVar("Data", Tuple[tf.Tensor, tf.Tensor], tf.Tensor)
DataPoint = tf.Tensor
MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]


class BayesianModel(Module):
    """ Bayesian model. """

    def neg_log_marginal_likelihood(self, *args, **kwargs) -> tf.Tensor:
        msg = (
            "`BayesianModel.neg_log_marginal_likelihood` is deprecated and "
            " and will be removed in a future release. Please update your code "
            " to use `BayesianModel.log_marginal_likelihood`."
        )
        warnings.warn(msg, category=DeprecationWarning)
        return -self.log_marginal_likelihood(*args, **kwargs)

    def log_marginal_likelihood(self, *args, **kwargs) -> tf.Tensor:
        return self.log_likelihood(*args, **kwargs) + self.log_prior()

    def log_prior(self) -> tf.Tensor:
        log_priors = [p.log_prior() for p in self.trainable_parameters]
        if log_priors:
            return tf.add_n(log_priors)
        else:
            return tf.convert_to_tensor(0.0, dtype=default_float())

    @abc.abstractmethod
    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
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
        self, Xnew: DataPoint, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError

    def predict_f_samples(
        self,
        Xnew: DataPoint,
        num_samples: Optional[int] = 1,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: DataPoint
            Input locations at which to draw samples
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.

        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )

        mean, cov = self.predict_f(Xnew, full_cov=full_cov)  # [N, P], [P, N, N] or [N, P]
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(
                mean_for_sample, cov, "full", num_samples=num_samples
            )  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            cov_structure = "full" if full_output_cov else "diag"
            samples = sample_mvn(
                mean, cov, cov_structure, num_samples=num_samples
            )  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def predict_y(
        self, Xnew: DataPoint, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_log_density(
        self, data: Data, full_cov: bool = False, full_output_cov: bool = False
    ):
        """
        Compute the log density of the data at the new data points.
        """
        X, Y = data
        f_mean, f_var = self.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_density(f_mean, f_var, Y)
