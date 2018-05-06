# Copyright 2016 the gpflow authors.
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

import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose
import pytest

import gpflow
from gpflow import settings
from gpflow.test_util import GPflowTestCase

class FlatModel(gpflow.models.Model):
    def _build_likelihood(self):
        return np.array(0., dtype=settings.float_type)

class TestPriorMode(GPflowTestCase):
    """
    these tests optimize the prior to find the mode numerically. Make sure the
    mode is the same as the known mode.
    """
    def prepare(self, autobuild=False):
        return FlatModel(autobuild=autobuild)

    def testGaussianMode(self):
        with self.test_context():
            m = self.prepare()
            m.x = gpflow.Param(1., autobuild=False)
            m.x.prior = gpflow.priors.Gaussian(3., 1.)

            m.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m)
            _ = [assert_allclose(v, 3) for v in m.read_trainables().values()]

    def testGaussianModeMatrix(self):
        with self.test_context():
            m = self.prepare()
            m.x = gpflow.Param(np.random.randn(4, 4), prior=gpflow.priors.Gaussian(-1., 10.))

            m.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m)
            _ = [assert_allclose(v, -1.) for v in m.read_trainables().values()]

    def testExponential(self):
        with self.test_context():
            m = self.prepare()
            m.x = gpflow.Param(1.0, prior=gpflow.priors.Exponential(1.0))
            m.compile()
            self.assertTrue(np.allclose(m.compute_log_prior(), -1.0))

        with self.test_context():
            m = self.prepare()
            m.x = gpflow.Param(1.0, prior=gpflow.priors.Exponential(2.0))
            m.compile()
            self.assertTrue(np.allclose(m.compute_log_prior(), np.log(2) - 2))

    def testGammaMode(self):
        with self.test_context():
            m = self.prepare()
            m.x = gpflow.Param(1.0, autobuild=False)
            shape, scale = 4., 5.
            m.x.prior = gpflow.priors.Gamma(shape, scale)

            m.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m)

            true_mode = (shape - 1.) * scale
            assert_allclose(m.x.read_value(), true_mode, 1e-3)

    def testLaplaceMode(self):
        with self.test_context():
            m = self.prepare()
            m.x = gpflow.Param(1.0, prior=gpflow.priors.Laplace(3., 10.))
            m.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m)
            _ = [assert_allclose(v, 3) for v in m.read_trainables().values()]

    def testLogNormalMode(self):
        with self.test_context():
            m = self.prepare()
            transform = gpflow.transforms.Exp()
            prior = gpflow.priors.LogNormal(3., 10.)
            m.x = gpflow.Param(1.0, prior=prior, transform=transform)
            m.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m)
            xmax = [transform.backward(x) for x in m.read_trainables().values()]
            assert_allclose(xmax, 3, rtol=1e4)

    def testBetaMode(self):
        with self.test_context():
            m = self.prepare()
            transform = gpflow.transforms.Logistic()
            m.x = gpflow.Param(0.1, prior=gpflow.priors.Beta(3., 3.), transform=transform)

            m.compile()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m)
            xmax = [transform.backward(x) for x in m.read_trainables().values()]
            assert_allclose(0.0, xmax, atol=1.e-5)

    def testUniform(self):
        with self.test_context():
            m = self.prepare()
            m.x = gpflow.Param(
                1.0, prior=gpflow.priors.Uniform(-2., 3.),
                transform=gpflow.transforms.Logistic(-2., 3.))

            m.compile()
            m.x = np.random.randn(1)[0]
            p1 = m.compute_log_prior()

            m.x = np.random.randn(1)[0]
            p2 = m.compute_log_prior()

            # prior should not be the same because a transformation has been applied.
            self.assertTrue(p1 != p2)


def mc_moments(prior, size=(10000, 1000)):
    np.random.seed(1)
    x = prior.sample(size)
    assert x.shape == size, "inline test that .sample() returns correct shape"
    mean = x.mean()
    var = x.var()
    # analytic moments are all np.atleast_1d() due to the parameters in gpflow.priors
    # hence we need to do the same for the MC moments so that the shapes match
    return np.atleast_1d(mean), np.atleast_1d(var)

def gaussian_moments(prior):
    return prior.mu, prior.var

def exponential_moments(prior):
    return 1 / prior.rate, prior.rate ** (-2)

def lognormal_moments(prior):
    mu, var = prior.mu, prior.var
    return np.exp(mu + var / 2), (np.exp(var) - 1) * np.exp(2 * mu + var)

def gamma_moments(prior):
    return prior.shape * prior.scale, prior.shape * prior.scale ** 2

def laplace_moments(prior):
    return prior.mu, 2 * prior.sigma ** 2

def beta_moments(prior):
    a, b = prior.a, prior.b
    return a / (a + b), (a * b) / ((a + b)**2 * (a + b + 1))

def uniform_moments(prior):
    a, b = prior.lower, prior.upper
    # this is the only prior that does not wrap parameters in np.atleast_1d()
    # so do it here to make sure all the shapes are consistent:
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return (a + b) / 2, (b - a)**2 / 12

@pytest.mark.parametrize("args", [
    ("Exponential", [1.3]),
    ("Gaussian", [-2.5, 3.4]),
    ("LogNormal", [-2.5, 1.4]),
    ("Gamma", [1.5, 0.7]),
    ("Laplace", [-2.5, 3.4]),
    ("Beta", [3.6, 0.4]),
    ("Uniform", [5.4, 8.9]),
    ])
def test_moments(args):
    classname, params = args
    cls = eval("gpflow.priors.{}".format(classname))
    prior = cls(*params)
    moments_func = eval("{}_moments".format(cls.__name__.lower()))
    rtol = 5e-3 if classname == "LogNormal" else 1e-3
    assert_allclose(moments_func(prior), mc_moments(prior), rtol=rtol,
                    err_msg="for {} prior".format(classname))

if __name__ == "__main__":
    tf.test.main()
