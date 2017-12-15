# Copyright 2017 the GPflow authors.
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
# limitations under the License.from __future__ import print_function


class ProbabilityDistribution:
    """
    This is the base class for a probability distributions,
    over which we take the expectations in the expectations framework.
    """
    pass


class Gaussian(ProbabilityDistribution):
    def __init__(self, mu, cov):
        self.mu = mu  # N x D
        self.cov = cov  # N x D x D


class DiagonalGaussian(ProbabilityDistribution):
    def __init__(self, mu, var):
        self.mu = mu  # N x D
        self.var = var  # N x D


class TimeSeriesGaussian(ProbabilityDistribution):
    def __init__(self, mu, cov):
        self.mu = mu  # N+1 x D
        self.cov = cov  # 2 x (N+1) x D x D