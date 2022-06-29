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


# Eventually, it would be nice to not have to have our own classes for
# probability distributions. The TensorFlow "distributions" framework would
# be a good replacement.
from abc import ABC, abstractmethod

from .base import TensorType
from .experimental.check_shapes import ErrorContext, Shape, check_shapes, register_get_shape


class ProbabilityDistribution(ABC):
    """
    This is the base class for a probability distributions,
    over which we take the expectations in the expectations framework.
    """

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """
        Return the shape of this distribution.

        Shape should be some variation of ``[N, D]``, where:

        * ``N`` is the number of data points.
        * ``D`` is the number of input dimensions.
        """


@register_get_shape(ProbabilityDistribution)
def get_probability_distribution_shape(
    shaped: ProbabilityDistribution, context: ErrorContext
) -> Shape:
    return shaped.shape


class Gaussian(ProbabilityDistribution):
    @check_shapes(
        "mu: [N, D]",
        "cov: [N, D, D]",
    )
    def __init__(self, mu: TensorType, cov: TensorType):
        self.mu = mu
        self.cov = cov

    @property
    def shape(self) -> Shape:
        return self.mu.shape  # type: ignore[no-any-return]


class DiagonalGaussian(ProbabilityDistribution):
    @check_shapes(
        "mu: [N, D]",
        "cov: [N, D]",
    )
    def __init__(self, mu: TensorType, cov: TensorType):
        self.mu = mu
        self.cov = cov

    @property
    def shape(self) -> Shape:
        return self.mu.shape  # type: ignore[no-any-return]


class MarkovGaussian(ProbabilityDistribution):
    """
    Gaussian distribution with Markov structure.
    Only covariances and covariances between t and t+1 need to be
    parameterised. We use the solution proposed by Carl Rasmussen, i.e. to
    represent
    Var[x_t] = cov[x_t, :, :] * cov[x_t, :, :].T
    Cov[x_t, x_{t+1}] = cov[t, :, :] * cov[t+1, :, :]
    """

    @check_shapes(
        "mu: [N_plus_1, D]",
        "cov: [2, N_plus_1, D, D]",
    )
    def __init__(self, mu: TensorType, cov: TensorType):
        self.mu = mu
        self.cov = cov

    @property
    def shape(self) -> Shape:
        shape = self.mu.shape
        if shape is None:
            return shape
        N_plus_1, D = shape
        N = N_plus_1 - 1
        return N, D
