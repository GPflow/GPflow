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
from typing import Optional, Tuple

import tensorflow as tf

from ..base import Parameter
from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction, Zero
from ..util import default_jitter, default_float

MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]


# class covstruct(Enum):
#     none = 0
#     diag = 1
#     full = 2
#     full_output = 3


class BayesianModel(tf.Module):
    """ Bayesian model. """

    def neg_log_marginal_likelihood(self, *args, **kwargs) -> tf.Tensor:
        return - tf.add(self.log_likelihood(*args, **kwargs), self.log_prior())

    def log_prior(self) -> tf.Tensor:
        if len(self.variables) == 0:
            return tf.convert_to_tensor(0., dtype=default_float())
        return tf.add_n([p.log_prior() for p in self.variables if isinstance(p, Parameter)])

    @abc.abstractmethod
    def log_likelihood(self, *args, **kwargs) -> tf.Tensor:
        pass


class GPModel(BayesianModel):
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

    def __init__(self,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None,
                 num_latent: int = 1,
                 seed: Optional[int] = None):
        super().__init__()
        self.num_latent = num_latent
        if mean_function is not None:
            self.num_latent = len(mean_function)
        else:
            mean_function = Zero()
        self.mean_function = mean_function
        self.kernel = kernel
        self.likelihood = likelihood

    @abc.abstractmethod
    def predict_f(self, X: tf.Tensor, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        pass

    def predict_f_samples(self, X, num_samples, full_cov=False, full_output_cov=False, jitter=None):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        jitter = default_jitter() if jitter is None else jitter
        mu, var = self.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)  # [N, P] or [P, N, N]
        jitter = tf.eye(mu.shape[0], dtype=X.dtype) * jitter
        samples = [None] * self.num_latent
        for i in range(self.num_latent):
            L = tf.linalg.cholesky(var[i, :, :] + jitter)
            shape = tf.stack([L.shape[0], num_samples])
            V = tf.random.normal(shape, dtype=L.dtype, seed=self.seed)
            samples[i] = mu[:, i:(i+1)] + L @ V
        return tf.linalg.transpose(tf.stack(samples))

    def predict_y(self, X):
        """
        Compute the mean and variance of held-out data at the points X
        """
        f_mean, f_var = self.predict_f(X)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_log_density(self, X, Y):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        f_mean, f_var = self.predict_f(X)
        return self.likelihood.predict_density(f_mean, f_var, Y)
