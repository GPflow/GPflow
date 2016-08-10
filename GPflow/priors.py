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


from . import densities
import tensorflow as tf
import numpy as np
from .param import Parameterized


class Prior(Parameterized):
    def logp(self, x):
        """
        The log density of the prior as x

        All priors (for the moment) are univariate, so if x is a vector or an
        array, this is the sum of the log densities.
        """
        raise NotImplementedError

    def __str__(self):
        """
        A short string to describe the prior at print time
        """
        raise NotImplementedError


class Gaussian(Prior):
    def __init__(self, mu, var):
        Prior.__init__(self)
        self.mu = np.atleast_1d(np.array(mu, np.float64))
        self.var = np.atleast_1d(np.array(var, np.float64))

    def logp(self, x):
        return tf.reduce_sum(densities.gaussian(x, self.mu, self.var))

    def __str__(self):
        return "N("+str(self.mu) + "," + str(self.var) + ")"


class LogNormal(Prior):
    def __init__(self, mu, var):
        Prior.__init__(self)
        self.mu = np.atleast_1d(np.array(mu, np.float64))
        self.var = np.atleast_1d(np.array(var, np.float64))

    def logp(self, x):
        return tf.reduce_sum(densities.lognormal(x, self.mu, self.var))

    def __str__(self):
        return "logN("+str(self.mu) + "," + str(self.var) + ")"


class Gamma(Prior):
    def __init__(self, shape, scale):
        Prior.__init__(self)
        self.shape = np.atleast_1d(np.array(shape, np.float64))
        self.scale = np.atleast_1d(np.array(scale, np.float64))

    def logp(self, x):
        return tf.reduce_sum(densities.gamma(self.shape, self.scale, x))

    def __str__(self):
        return "Ga("+str(self.shape) + "," + str(self.scale) + ")"


class Laplace(Prior):
    def __init__(self, mu, sigma):
        Prior.__init__(self)
        self.mu = np.atleast_1d(np.array(mu, np.float64))
        self.sigma = np.atleast_1d(np.array(sigma, np.float64))

    def logp(self, x):
        return tf.reduce_sum(densities.laplace(self.mu, self.sigma, x))

    def __str__(self):
        return "Lap.("+str(self.mu) + "," + str(self.sigma) + ")"


class Uniform(Prior):
    def __init__(self, lower=0, upper=1):
        self.log_height = - np.log(upper - lower)
        self.lower, self.upper = lower, upper

    def logp(self, x):
        return self.log_height * tf.cast(tf.size(x), tf.float64)

    def __str__(self):
        return "U("+str(self.lower) + "," + str(self.upper) + ")"
