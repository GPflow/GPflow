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
import pytest
import scipy.stats
from numpy.random import randn
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal as mvn

from gpflow import logdensities
from gpflow.base import AnyNDArray
from gpflow.utilities import to_default_float

rng = np.random.RandomState(1)


@pytest.mark.parametrize("x, mu, var", [(0.9, 0.5, 1.3)])
def test_gaussian(x: float, mu: float, var: float) -> None:
    gpf = logdensities.gaussian(x, mu, var).numpy()
    sps = scipy.stats.norm(loc=mu, scale=np.sqrt(var)).logpdf(x)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize("x, mu, var", [(0.9, 0.5, 1.3)])
def test_lognormal(x: float, mu: float, var: float) -> None:
    gpf = logdensities.lognormal(x, mu, var).numpy()
    sps = scipy.stats.lognorm(s=np.sqrt(var), scale=np.exp(mu)).logpdf(x)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize(
    "x, p",
    [
        [1, 0.6],
        [0, 0.6],
    ],
)
def test_bernoulli(x: float, p: float) -> None:
    gpf = logdensities.bernoulli(x, p).numpy()
    sps = scipy.stats.bernoulli.logpmf(k=x, p=p)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize(
    "x, lam",
    [
        [0, 1.3],
        [1, 1.3],
        [2, 1.3],
    ],
)
def test_poisson(x: float, lam: float) -> None:
    gpf = logdensities.poisson(x, lam).numpy()
    sps = scipy.stats.poisson.logpmf(k=x, mu=lam)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize("x, scale", [(0.9, 1.3)])
def test_exponential(x: float, scale: float) -> None:
    gpf = logdensities.exponential(x, scale).numpy()
    sps = scipy.stats.expon(loc=0.0, scale=scale).logpdf(x)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize("x, shape, scale", [(0.9, 0.5, 1.3)])
def test_gamma(x: float, shape: float, scale: float) -> None:
    gpf = logdensities.gamma(x, shape, scale).numpy()
    sps = scipy.stats.gamma(a=shape, loc=0.0, scale=scale).logpdf(x)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize(
    "x, mean, scale, df",
    [
        (0.9, 0.5, 1.3, 1),
        (0.9, 0.5, 1.3, 2),
        (0.9, 0.5, 1.3, 3),
    ],
)
def test_student_t(x: float, mean: float, scale: float, df: int) -> None:
    cast = to_default_float
    gpf = logdensities.student_t(cast(x), cast(mean), cast(scale), df).numpy()
    sps = scipy.stats.t(df=df, loc=mean, scale=scale).logpdf(x)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize("x, alpha, beta", [(0.9, 0.5, 1.3)])
def test_beta(x: float, alpha: float, beta: float) -> None:
    gpf = logdensities.beta(x, alpha, beta).numpy()
    sps = scipy.stats.beta(a=alpha, b=beta).logpdf(x)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize("x, mu, sigma", [(0.9, 0.5, 1.3)])
def test_laplace(x: float, mu: float, sigma: float) -> None:
    gpf = logdensities.laplace(x, mu, sigma).numpy()
    sps = scipy.stats.laplace(loc=mu, scale=sigma).logpdf(x)
    np.testing.assert_allclose(gpf, sps)


@pytest.mark.parametrize("x", [randn(4, 10), randn(4, 1)])
@pytest.mark.parametrize("mu", [randn(4, 10), randn(4, 1)])
@pytest.mark.parametrize("cov_sqrt", [randn(4, 4), np.eye(4)])
def test_multivariate_normal(x: AnyNDArray, mu: AnyNDArray, cov_sqrt: AnyNDArray) -> None:
    cov = np.dot(cov_sqrt, cov_sqrt.T)
    L = np.linalg.cholesky(cov)

    gp_result = logdensities.multivariate_normal(x, mu, L)

    if mu.shape[1] > 1:
        if x.shape[1] > 1:
            sp_result = [mvn.logpdf(x[:, i], mu[:, i], cov) for i in range(mu.shape[1])]
        else:
            sp_result = [mvn.logpdf(x.ravel(), mu[:, i], cov) for i in range(mu.shape[1])]
    else:
        sp_result = mvn.logpdf(x.T, mu.ravel(), cov)
    assert_allclose(gp_result, sp_result)
