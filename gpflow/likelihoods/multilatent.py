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

import abc
from typing import Callable, Optional, Type

import tensorflow as tf
import tensorflow_probability as tfp

from ..utilities import positive
from .base import QuadratureLikelihood


class MultiLatentLikelihood(QuadratureLikelihood):
    r"""
    A Likelihood which assumes that a single dimensional observation is driven
    by multiple latent GPs.

    Note that this implementation does not allow for taking into account
    covariance between outputs.
    """

    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(
            latent_dim=latent_dim, observation_dim=1, **kwargs,
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
        **kwargs,
    ):
        """
        :param latent_dim: number of arguments to the `conditional_distribution` callable
        :param conditional_distribution: function from Fs to a tfp Distribution,
            where Fs has shape [..., latent_dim]
        """
        super().__init__(latent_dim, **kwargs)
        self.conditional_distribution = conditional_distribution

    def _log_prob(self, Fs, Y) -> tf.Tensor:
        """
        The log probability density log p(Y|F)

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., 1]:
        :returns: log pdf, with shape [...]
        """
        return tf.squeeze(self.conditional_distribution(Fs).log_prob(Y), -1)

    def _conditional_mean(self, Fs):
        """
        The conditional marginal mean of Y|F: [E(Y₁|F)]

        :param Fs: function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean [..., 1]
        """
        return self.conditional_distribution(Fs).mean()

    def _conditional_variance(self, Fs):
        """
        The conditional marginal variance of Y|F: [Var(Y₁|F)]

        :param Fs: function evaluation Tensor, with shape [..., latent_dim]
        :returns: variance [..., 1]
        """
        return self.conditional_distribution(Fs).variance()


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
        **kwargs,
    ):
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

        def conditional_distribution(Fs) -> tfp.distributions.Distribution:
            tf.debugging.assert_equal(tf.shape(Fs)[-1], 2)
            loc = Fs[..., :1]
            scale = self.scale_transform(Fs[..., 1:])
            return distribution_class(loc, scale)

        super().__init__(
            latent_dim=2, conditional_distribution=conditional_distribution, **kwargs,
        )
