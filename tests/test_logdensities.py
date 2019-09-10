# Copyright 2018 the GPflow authors.
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
from numpy.random import randn
import pytest
from gpflow import logdensities
from scipy.stats import multivariate_normal as mvn
from numpy.testing import assert_allclose

rng = np.random.RandomState(1)


@pytest.mark.parametrize("x", [randn(4, 10), randn(4, 1)])
@pytest.mark.parametrize("mu", [randn(4, 10), randn(4, 1)])
@pytest.mark.parametrize("cov_sqrt", [randn(4, 4), np.eye(4)])
def test_multivariate_normal(x, mu, cov_sqrt):
    cov = np.dot(cov_sqrt, cov_sqrt.T)
    L = np.linalg.cholesky(cov)

    gp_result = logdensities.multivariate_normal(x, mu, L)

    if mu.shape[1] > 1:
        if x.shape[1] > 1:
            sp_result = [
                mvn.logpdf(x[:, i], mu[:, i], cov) for i in range(mu.shape[1])
            ]
        else:
            sp_result = [
                mvn.logpdf(x.ravel(), mu[:, i], cov)
                for i in range(mu.shape[1])
            ]
    else:
        sp_result = mvn.logpdf(x.T, mu.ravel(), cov)
    assert_allclose(gp_result, sp_result)
