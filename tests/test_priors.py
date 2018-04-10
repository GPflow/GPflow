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
# limitations under the License.from __future__ import print_function

import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose

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

            # prior should no be the same because a transformation has been applied.
            self.assertTrue(p1 != p2)


if __name__ == "__main__":
    tf.test.main()
