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

import abc
from typing import Optional, Tuple

import tensorflow as tf

from ..base import Module
from ..conditionals.util import sample_mvn
from ..kernels import Kernel, MultioutputKernel
from ..likelihoods import Likelihood, SwitchedLikelihood
from ..mean_functions import MeanFunction, Zero
from ..utilities import to_default_float
from .training_mixins import InputData, RegressionData

MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]


class BayesianModel(Module, metaclass=abc.ABCMeta):
    """ Bayesian model. """

    def log_prior_density(self) -> tf.Tensor:
        """
        Sum of the log prior probability densities of all (constrained) variables in this model.
        """
        if self.trainable_parameters:
            return tf.add_n([p.log_prior_density() for p in self.trainable_parameters])
        else:
            return to_default_float(0.0)

    def log_posterior_density(self, *args, **kwargs) -> tf.Tensor:
        """
        This may be the posterior with respect to the hyperparameters (e.g. for
        GPR) or the posterior with respect to the function (e.g. for GPMC and
        SGPMC). It assumes that maximum_log_likelihood_objective() is defined
        sensibly.
        """
        return self.maximum_log_likelihood_objective(*args, **kwargs) + self.log_prior_density()

    def _training_loss(self, *args, **kwargs) -> tf.Tensor:
        """
        Training loss definition. To allow MAP (maximum a-posteriori) estimation,
        adds the log density of all priors to maximum_log_likelihood_objective().
        """
        return -(self.maximum_log_likelihood_objective(*args, **kwargs) + self.log_prior_density())

    @abc.abstractmethod
    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.Tensor:
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
        num_latent_gps: int = None,
    ):
        super().__init__()
        assert num_latent_gps is not None, "GPModel requires specification of num_latent_gps"
        self.num_latent_gps = num_latent_gps
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.kernel = kernel
        self.likelihood = likelihood

    @staticmethod
    def calc_num_latent_gps_from_data(data, kernel: Kernel, likelihood: Likelihood) -> int:
        """
        Calculates the number of latent GPs required based on the data as well
        as the type of kernel and likelihood.
        """
        _, Y = data
        output_dim = Y.shape[-1]
        return GPModel.calc_num_latent_gps(kernel, likelihood, output_dim)

    @staticmethod
    def calc_num_latent_gps(kernel: Kernel, likelihood: Likelihood, output_dim: int) -> int:
        """
        Calculates the number of latent GPs required given the number of
        outputs `output_dim` and the type of likelihood and kernel.

        Note: It's not nice for `GPModel` to need to be aware of specific
        likelihoods as here. However, `num_latent_gps` is a bit more broken in
        general, we should fix this in the future. There are also some slightly
        problematic assumptions re the output dimensions of mean_function.
        See https://github.com/GPflow/GPflow/issues/1343
        """
        if isinstance(kernel, MultioutputKernel):
            # MultioutputKernels already have num_latent_gps attributes
            num_latent_gps = kernel.num_latent_gps
        elif isinstance(likelihood, SwitchedLikelihood):
            # the SwitchedLikelihood partitions/stitches based on the last
            # column in Y, but we should not add a separate latent GP for this!
            # hence decrement by 1
            num_latent_gps = output_dim - 1
            assert num_latent_gps > 0
        else:
            num_latent_gps = output_dim

        return num_latent_gps

    @abc.abstractmethod
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        raise NotImplementedError

    def predict_f_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        full_cov: bool = True,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.

        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
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

        # check below for shape info
        mean, cov = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(
                mean_for_sample, cov, full_cov, num_samples=num_samples
            )  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            samples = sample_mvn(
                mean, cov, full_output_cov, num_samples=num_samples
            )  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_log_density(
        self, data: RegressionData, full_cov: bool = False, full_output_cov: bool = False
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        X, Y = data
        f_mean, f_var = self.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)
