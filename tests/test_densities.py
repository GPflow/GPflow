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
# limitations under the License.from __future__ import print_function

import numpy as np
from numpy.random import randn
import tensorflow as tf
import pytest
import gpflow
from gpflow import densities
from gpflow.test_util import session_tf
from scipy.stats import multivariate_normal as mvn
from numpy.testing import assert_allclose


rng = np.random.RandomState(1)


@pytest.mark.parametrize("x", [randn(4,10), randn(4,1), randn(4)])
@pytest.mark.parametrize("mu", [randn(4,10), randn(4,1), randn(4)])
@pytest.mark.parametrize("cov", [randn(4,4), np.eye(4)])
def test_multivariate_normal(session_tf, x, mu, cov):
    cov = np.dot(cov, cov.T)
    L = np.linalg.cholesky(cov)
    gp_result = densities.multivariate_normal(x, mu, L)
    gp_result = session_tf.run(gp_result)
    if len(mu.shape) > 1 and mu.shape[1] > 1:
        if len(x.shape) > 1 and x.shape[1] > 1:
            sp_result = [mvn.logpdf(x[:,i], mu[:,i], cov) for i in range(mu.shape[1])]
        else:
            sp_result = [mvn.logpdf(x.ravel(), mu[:, i], cov) for i in range(mu.shape[1])]
    else:
        sp_result = mvn.logpdf(x.T, mu.ravel(), cov)
    assert_allclose(gp_result, sp_result)
