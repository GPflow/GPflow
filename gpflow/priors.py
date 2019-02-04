# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews
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
"""
For a fully Bayesian treatment of a model it is necessary to assign prior
distributions over unknown parameters. Ideally the uncertainty of these
parameters in the posterior should also be utilised when making predictions.

GPflow allows priors to be assigned to any parameter, such as kernel
hyperparameters, likelihood parameters, model parameters, mean function
parameters, etc. If prior information is available about the likely values of
these parameters, this can often aid optimisation.

Many default GPflow models only allow Maximum a Posteriori (MAP) estimation of
these parameters (the most likely value in the posterior is used during
prediction, but the uncertainty around this value is not used). However,
assigning prior distributions for unknown values can still be useful in this
situation.

MCMC based inference models are the exception, here the uncertainty in the
parameters is approximately integrated over during predictions, using samples.
See the `MCMC notebook <notebooks/mcmc.html>`_ for further details on using MCMC
models in GPflow.

GPflow provides a range of standard prior distributions to be used, all being
subclasses of :class:`Prior <gpflow.priors.Prior>`. Each distribution makes
different assumptions about the distribution of likely values and contains more
or less support for different parameter values that it is assigned to depending
on both the distribution type and their own (fixed) hyperparameters.

For example is a value is known to only be positive, a prior distribution with
only positive support should be chosen, such as a LogNormal prior. If a value
should only be between 0 and 1, a Beta distribution may be used.
"""

import tensorflow as tf
import numpy as np

from . import logdensities
from . import settings

from .params import Parameterized
from .core.base import IPrior


class Prior(Parameterized, IPrior):  # pylint: disable=W0223
    pass


class Exponential(Prior):
    """
    Exponential distribution, parameterised in terms of its rate
    (inverse scale).

    Support: [0, inf)

    Note, the rate parameter may be a vector, this assumes a different
    rate parameter per dimension of the parameter the prior is over.
    """

    def __init__(self, rate):
        """
        :param float rate: Rate parameter (inverse scale) (rate > 0)
        """
        Prior.__init__(self)
        self.rate = np.atleast_1d(np.array(rate, settings.float_type))
        if any(self.rate <= 0):  # pragma: no cover
            raise ValueError("The rate parameter has to be positive.")

    def logp(self, x):
        scale = 1 / self.rate
        return tf.reduce_sum(logdensities.exponential(x, scale))

    def sample(self, shape=(1,)):
        return np.random.exponential(scale=1 / self.rate, size=shape)

    def __str__(self):
        return "Exp({})".format(self.rate.squeeze())


class Gaussian(Prior):
    """
    Gaussian distribution, parameterised in terms of its mean and variance

    Support: (-inf, inf)

    Note, the mean and variance parameters may be vectors, this assumes a
    different univariate Gaussian per dimension of the parameter the prior is
    over. The variance parameter must also be positive.
    """
    def __init__(self, mu, var):
        """
        :param float mu: mean parameter of Gaussian
        :param float var: variance parameter of Gaussian (var > 0)
        """
        Prior.__init__(self)
        self.mu = np.atleast_1d(np.array(mu, settings.float_type))
        self.var = np.atleast_1d(np.array(var, settings.float_type))
        if any(self.var <= 0):  # pragma: no cover
            raise ValueError("The var parameter has to be positive.")

    def logp(self, x):
        return tf.reduce_sum(logdensities.gaussian(x, self.mu, self.var))

    def sample(self, shape=(1,)):
        return self.mu + np.sqrt(self.var) * np.random.randn(*shape)

    def __str__(self):
        return "N({},{})".format(self.mu.squeeze(), self.var.squeeze())


class LogNormal(Prior):
    """
    Log-normal distribution, parameterised in terms of its mean and variance

    Support: [0, inf)

    Note, the mean and variance parameters may be vectors, this assumes a
    different univariate log-normal per dimension of the parameter the prior is
    over. The variance parameter must also be positive.
    """
    def __init__(self, mu, var):
        """
        :param float mu: mean parameter of log-normal
        :param float var: variance parameter of log-normal (var > 0)
        """
        Prior.__init__(self)
        self.mu = np.atleast_1d(np.array(mu, settings.float_type))
        self.var = np.atleast_1d(np.array(var, settings.float_type))
        if any(self.var <= 0):  # pragma: no cover
            raise ValueError("The var parameter has to be positive.")

    def logp(self, x):
        return tf.reduce_sum(logdensities.lognormal(x, self.mu, self.var))

    def sample(self, shape=(1,)):
        return np.exp(self.mu + np.sqrt(self.var) * np.random.randn(*shape))

    def __str__(self):
        return "logN({},{})".format(self.mu.squeeze(), self.var.squeeze())


class Gamma(Prior):
    """
    Gamma distribution, parameterised in terms of its shape and scale

    Support: [0, inf)

    Note, the shape and scale parameters may be vectors, this assumes a
    different univariate Gammma distribution per dimension of the parameter the
    prior is over. Both shape and scale parameters must be positive
    """
    def __init__(self, shape, scale):
        """
        :param float shape: shape parameter of Gamma distribution (shape > 0)
        :param float scale: scale parameter of Gamma distribution (scale > 0)
        """
        Prior.__init__(self)
        self.shape = np.atleast_1d(np.array(shape, settings.float_type))
        self.scale = np.atleast_1d(np.array(scale, settings.float_type))
        if any(self.shape <= 0):  # pragma: no cover
            raise ValueError("The shape parameter has to be positive.")
        if any(self.scale <= 0):  # pragma: no cover
            raise ValueError("The scale parameter has to be positive.")

    def logp(self, x):
        return tf.reduce_sum(logdensities.gamma(x, self.shape, self.scale))

    def sample(self, shape=(1,)):
        return np.random.gamma(self.shape, self.scale, size=shape)

    def __str__(self):
        return "Ga({},{})".format(self.shape.squeeze(), self.scale.squeeze())


class Laplace(Prior):
    """
    Laplace distribution, parameterised in terms of its mu (shape) and sigma
    (scale)

    Support: (-inf, inf)

    Note, the mean and scale parameters may be vectors, this assumes a
    different univariate Laplace distribution per dimension of the parameter
    the prior is over. The scale parameter must be positive.
    """
    def __init__(self, mu, sigma):
        """
        :param float mu: shape parameter of Laplace distribution
        :param float sigma: scale parameter of Laplace distribution (sigma > 0)
        """
        Prior.__init__(self)
        self.mu = np.atleast_1d(np.array(mu, settings.float_type))
        self.sigma = np.atleast_1d(np.array(sigma, settings.float_type))
        if any(self.sigma <= 0):  # pragma: no cover
            raise ValueError("The sigma parameter has to be positive.")

    def logp(self, x):
        return tf.reduce_sum(logdensities.laplace(x, self.mu, self.sigma))

    def sample(self, shape=(1,)):
        return np.random.laplace(self.mu, self.sigma, size=shape)

    def __str__(self):
        return "Lap.({},{})".format(self.mu.squeeze(), self.sigma.squeeze())


class Beta(Prior):
    """
    Beta distribution, parameterised in terms of its 'a' (shape) and 'b'
    (scale) parameters

    Support: [0, 1]

    Note, the shape and scale parameters may be vectors, this assumes a
    different univariate Beta distribution per dimension of the parameter
    the prior is over. Both parameters must be positive.
    """
    def __init__(self, a, b):
        """
        :param float a: shape parameter of Laplace distribution (a > 0)
        :param float b: scale parameter of Laplace distribution (b > 0)
        """
        Prior.__init__(self)
        self.a = np.atleast_1d(np.array(a, settings.float_type))
        self.b = np.atleast_1d(np.array(b, settings.float_type))
        if any(self.a <= 0) or any(self.b <= 0):  # pragma: no cover
            raise ValueError("The parameters have to be positive.")

    def logp(self, x):
        return tf.reduce_sum(logdensities.beta(x, self.a, self.b))

    def sample(self, shape=(1,)):
        return np.random.beta(self.a, self.b, size=shape)

    def __str__(self):
        return "Beta({},{})".format(self.a.squeeze(), self.b.squeeze())


class Uniform(Prior):
    """
    Uniform distribution, parameterised in terms of its lower and upper

    Support: [lower, upper]
    """
    def __init__(self, lower=0., upper=1.):
        """
        :param float lower: lower possible value of the uniform distribution
        :param float upper: upper possible value of the uniform distribution
        """
        Prior.__init__(self)
        self.lower, self.upper = lower, upper
        if lower >= upper:  # pragma: no cover
            raise ValueError("The lower bound has to be smaller than the upper bound.")

    @property
    def log_height(self):
        return - np.log(self.upper - self.lower)

    def logp(self, x):
        return self.log_height * tf.cast(tf.size(x), settings.float_type)

    def sample(self, shape=(1,)):
        return (self.lower +
                (self.upper - self.lower)*np.random.rand(*shape))

    def __str__(self):
        return "U({},{})".format(self.lower, self.upper)

