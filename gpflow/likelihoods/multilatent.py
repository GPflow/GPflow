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
from typing import Callable, Type, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from ..utilities import positive
from .base import QuadratureLikelihood


# NOTE- in the following we're assuming outputs are independent, i.e. full_output_cov=False


class MultiLatentLikelihood(QuadratureLikelihood):
    r"""
    A Likelihood which assumes that a single dimensional observation is driven
    by multiple latent GPs.
    """

    def __init__(self, latent_dim: int, *, num_gauss_hermite_points: int = 21):
        # TODO: use same variable name for num_gauss_hermite_points as in ScalarLikelihood.
        super().__init__(latent_dim=latent_dim, observation_dim=1, n_gh=num_gauss_hermite_points)


class MultiLatentTFPConditional(MultiLatentLikelihood):
    """
    MultiLatent likelihood where the conditional distribution
    is given by a TensorFlow Probability Distribution.
    """

    def __init__(
        self,
        latent_dim: int,
        conditional_distribution: Callable[..., tfp.distributions.Distribution],
        num_gauss_hermite_points: int = 21,
    ):
        """
        :param latent_dim: number of arguments to the `conditional_distribution` callable
        :param conditional_distribution: function from Fs to a tfp Distribution,
            where Fs has shape [..., latent_dim]
        """
        super().__init__(latent_dim, num_gauss_hermite_points=num_gauss_hermite_points)
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
        transform: tfp.bijectors.Bijector = positive(base="exp"),
        num_gauss_hermite_points: int = 21,
    ):
        """
        :param distribution_class: distribution class parameterized by `loc` and `scale`
            as first and second argument, respectivily.
        :param transform: callable applied to the variance GP to make it positive.
            Typically, `tf.exp` or `tf.softplus`, but can be any function f: R -> R^+.
        """

        def conditional_distribution(Fs) -> tfp.distributions.Distribution:
            tf.debugging.assert_equal(tf.shape(Fs)[-1], 2)
            loc = Fs[..., :1]
            scale = transform(Fs[..., 1:] / 2)
            return distribution_class(loc, scale)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            num_gauss_hermite_points=num_gauss_hermite_points,
        )
