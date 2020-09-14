# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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

from .. import logdensities
from ..base import Parameter
from ..utilities import positive
from .base import ScalarLikelihood
from .utils import inv_probit


class Gaussian(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self, variance=1.0, variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs):
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if variance <= variance_lower_bound:
            raise ValueError(
                f"The variance of the Gaussian likelihood must be strictly greater than {variance_lower_bound}"
            )

        self.variance = Parameter(variance, transform=positive(lower=variance_lower_bound))

    def _scalar_log_prob(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance,
            axis=-1,
        )


class Exponential(ScalarLikelihood):
    def __init__(self, invlink=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink

    def _scalar_log_prob(self, F, Y):
        return logdensities.exponential(Y, self.invlink(F))

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        return tf.square(self.invlink(F))

    def _variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return tf.reduce_sum(-tf.exp(-Fmu + Fvar / 2) * Y - Fmu, axis=-1)
        return super()._variational_expectations(Fmu, Fvar, Y)


class StudentT(ScalarLikelihood):
    def __init__(self, scale=1.0, df=3.0, **kwargs):
        """
        :param scale float: scale parameter
        :param df float: degrees of freedom
        """
        super().__init__(**kwargs)
        self.df = df
        self.scale = Parameter(scale, transform=positive())

    def _scalar_log_prob(self, F, Y):
        return logdensities.student_t(Y, F, self.scale, self.df)

    def _conditional_mean(self, F):
        return F

    def _conditional_variance(self, F):
        var = (self.scale ** 2) * (self.df / (self.df - 2.0))
        return tf.fill(tf.shape(F), tf.squeeze(var))


class Gamma(ScalarLikelihood):
    """
    Use the transformed GP to give the *scale* (inverse rate) of the Gamma
    """

    def __init__(self, invlink=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink
        self.shape = Parameter(1.0, transform=positive())

    def _scalar_log_prob(self, F, Y):
        return logdensities.gamma(Y, self.shape, self.invlink(F))

    def _conditional_mean(self, F):
        return self.shape * self.invlink(F)

    def _conditional_variance(self, F):
        scale = self.invlink(F)
        return self.shape * (scale ** 2)

    def _variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return tf.reduce_sum(
                -self.shape * Fmu
                - tf.math.lgamma(self.shape)
                + (self.shape - 1.0) * tf.math.log(Y)
                - Y * tf.exp(-Fmu + Fvar / 2.0),
                axis=-1,
            )
        else:
            return super()._variational_expectations(Fmu, Fvar, Y)


class Beta(ScalarLikelihood):
    """
    This uses a reparameterisation of the Beta density. We have the mean of the
    Beta distribution given by the transformed process:

        m = invlink(f)

    and a scale parameter. The familiar α, β parameters are given by

        m     = α / (α + β)
        scale = α + β

    so:
        α = scale * m
        β  = scale * (1-m)
    """

    def __init__(self, invlink=inv_probit, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = Parameter(scale, transform=positive())
        self.invlink = invlink

    def _scalar_log_prob(self, F, Y):
        mean = self.invlink(F)
        alpha = mean * self.scale
        beta = self.scale - alpha
        return logdensities.beta(Y, alpha, beta)

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        mean = self.invlink(F)
        return (mean - tf.square(mean)) / (self.scale + 1.0)
