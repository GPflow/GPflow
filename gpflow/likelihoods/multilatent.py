# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Any, Callable, Optional, Type

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes, inherit_check_shapes

from ..base import TensorType
from ..utilities import positive
from .base import QuadratureLikelihood


class MultiLatentLikelihood(QuadratureLikelihood):
    r"""
    A Likelihood which assumes that a single dimensional observation is driven
    by multiple latent GPs.

    Note that this implementation does not allow for taking into account
    covariance between outputs.
    """

    def __init__(self, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(
            input_dim=None,
            latent_dim=latent_dim,
            observation_dim=1,
            **kwargs,
        )


class MultiLatentTFPConditional(MultiLatentLikelihood):
    """
    MultiLatent likelihood where the conditional distribution
    is given by a TensorFlow Probability Distribution.
    """

    def __init__(
        self,
        latent_dim: int,
        conditional_distribution: Callable[..., tfp.distributions.Distribution],
        **kwargs: Any,
    ):
        """
        :param latent_dim: number of arguments to the `conditional_distribution` callable
        :param conditional_distribution: function from F to a tfp Distribution,
            where F has shape [..., latent_dim]
        """
        super().__init__(latent_dim, **kwargs)
        self.conditional_distribution = conditional_distribution

    @inherit_check_shapes
    def _log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        """
        The log probability density log p(Y|F)

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., 1]:
        :returns: log pdf, with shape [...]
        """
        return tf.squeeze(self.conditional_distribution(F).log_prob(Y), -1)

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        """
        The conditional marginal mean of Y|F: [E(Y₁|F)]

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean [..., 1]
        """
        return self.conditional_distribution(F).mean()

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        """
        The conditional marginal variance of Y|F: [Var(Y₁|F)]

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :returns: variance [..., 1]
        """
        return self.conditional_distribution(F).variance()


class HeteroskedasticTFPConditional(MultiLatentTFPConditional):
    """
    Heteroskedastic Likelihood where the conditional distribution
    is given by a TensorFlow Probability Distribution.
    The `loc` and `scale` of the distribution are given by a
    two-dimensional multi-output GP.
    """

    def __init__(
        self,
        distribution_class: Type[tfp.distributions.Distribution] = tfp.distributions.Normal,
        scale_transform: Optional[tfp.bijectors.Bijector] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param distribution_class: distribution class parameterized by `loc` and `scale`
            as first and second argument, respectively.
        :param scale_transform: callable/bijector applied to the latent
            function modelling the scale to ensure its positivity.
            Typically, `tf.exp` or `tf.softplus`, but can be any function f: R -> R^+. Defaults to exp if not explicitly specified.
        """
        if scale_transform is None:
            scale_transform = positive(base="exp")
        self.scale_transform = scale_transform

        @check_shapes(
            "F: [batch..., 2]",
        )
        def conditional_distribution(F: TensorType) -> tfp.distributions.Distribution:
            loc = F[..., :1]
            scale = self.scale_transform(F[..., 1:])
            return distribution_class(loc, scale)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )


class HeteroskedasticGaussian(QuadratureLikelihood):
    """
    Analytical mplementation of the Heteroskedastic Gaussian Likelihood
    with exponential function as the scale transform function.
    The variational_expectations and predict_mean_and_var methods have analytical
    expressions do not use quadrature.
    The predict_log_density method is implemented with an analytical marginalization
    of the location-modeling GP and uses a one-dimensional quadrature to marginalize 
    the scale-modeling GP.
    """

    def __init__(self, quadrature=None):
        # TODO: This should be tested whenever the quadrature attirbute is set.
        # Maybe this should be done in the base class instead?
        if quadrature and quadrature.dim != self._quadrature_dim:
            raise Exception("If passing quadrature, quadrature.dim must be 1")

        super().__init__(observation_dim=1, latent_dim=2, quadrature=quadrature)
        self.scale_transform = tfp.bijectors.Exp()

    @property
    def _quadrature_dim(self) -> int:
        """
        By default, this returns self.latent_dim. However, in this class, the 
        location-modeling GP's is marginalized analytically. Hence, this is overriden
        to 1 as the quadrature only works with the scale-modeling GP.        
        """
        return 1

    def _loc(self, F):
        f = F[..., 0, None]
        loc = f
        return loc

    def _scale(self, F):
        g = F[..., 1, None]
        scale = tf.exp(g)
        return scale

    def _split_mean_and_var(self, Fmu, Fvar):
        m_f = Fmu[..., 0, None]
        m_g = Fmu[..., 1, None]

        k_f = Fvar[..., 0, None]
        k_g = Fvar[..., 1, None]

        return m_f, k_f, m_g, k_g

    def _conditional_mean(self, F):
        return self._loc(F)

    def _conditional_variance(self, F):
        return self._scale(F) ** 2

    def _log_prob(self, F, Y):
        y_given_fg = tfp.distributions.Normal(loc=self._loc(F), scale=self._scale(F))
        return y_given_fg.log_prob(Y)

    def _predict_mean_and_var(self, Fmu, Fvar):
        m_f, k_f, m_g, k_g = self._split_mean_and_var(Fmu, Fvar)
        Ymean = m_f
        Yvar = k_f + tf.exp(2 * m_g + 2 * k_g)
        return Ymean, Yvar

    def _predict_log_density(self, Fmu, Fvar, Y):
        m_f, k_f, m_g, k_g = self._split_mean_and_var(Fmu, Fvar)

        def log_prob(g):
            y_given_g = tfp.distributions.Normal(loc=m_f, scale=tf.sqrt(k_f + tf.exp(2 * g)))
            return y_given_g.log_prob(Y)

        result = self.quadrature(log_prob, m_g, k_g)
        return tf.squeeze(result, axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        m_f, k_f, m_g, k_g = self._split_mean_and_var(Fmu, Fvar)

        result = -0.5 * (
            np.log(2 * np.pi) + 2 * m_g + ((Y - m_f) ** 2 + k_f) * tf.exp(2 * k_g - 2 * m_g)
        )
        return tf.squeeze(result, axis=-1)
