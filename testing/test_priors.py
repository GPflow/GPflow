# Copyright 2016 the GPflow authors.
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

import unittest
import gpflow
import numpy as np
import tensorflow as tf


class PriorModeTests(unittest.TestCase):
    """
    these tests optimize the prior to find the mode numerically. Make sure the
    mode is the same as the known mode.
    """
    def setUp(self):
        tf.reset_default_graph()

        class FlatModel(gpflow.model.Model):
            def build_likelihood(self):
                return 0
        self.m = FlatModel()

    def testGaussianMode(self):
        self.m.x = gpflow.param.Param(1.0)
        self.m.x.prior = gpflow.priors.Gaussian(3, 1)
        self.m.optimize(disp=0)

        xmax = self.m.get_free_state()
        self.assertTrue(np.allclose(xmax, 3))

    def testGaussianModeMatrix(self):
        self.m.x = gpflow.param.Param(np.random.randn(4, 4))
        self.m.x.prior = gpflow.priors.Gaussian(-1, 10)
        self.m.optimize(disp=0)

        xmax = self.m.get_free_state()
        self.assertTrue(np.allclose(xmax, -1))

    def testGammaMode(self):
        self.m.x = gpflow.param.Param(1.0)
        shape, scale = 4., 5.
        self.m.x.prior = gpflow.priors.Gamma(shape, scale)
        self.m.optimize(disp=0)

        true_mode = (shape - 1.) * scale
        self.assertTrue(np.allclose(self.m.x.value, true_mode, 1e-3))

    def testLaplaceMode(self):
        self.m.x = gpflow.param.Param(1.0)
        self.m.x.prior = gpflow.priors.Laplace(3, 10)
        self.m.optimize(disp=0)

        xmax = self.m.get_free_state()
        self.assertTrue(np.allclose(xmax, 3))

    def testLogNormalMode(self):
        self.m.x = gpflow.param.Param(1.0)
        self.m.x.prior = gpflow.priors.LogNormal(3, 10)
        self.m.x.transform = gpflow.transforms.Exp()
        self.m.optimize(disp=0)

        xmax = self.m.get_free_state()
        self.assertTrue(np.allclose(xmax, 3))

    def testBetaMode(self):
        self.m.x = GPflow.param.Param(0.1)
        self.m.x.prior = GPflow.priors.Beta(3., 3.)
        self.m.x.transform = GPflow.transforms.Logistic()

        self.m.optimize(disp=0, tol=1e-8)

        xmax = self.m.get_free_state()
        self.assertTrue(np.allclose(0.0, xmax))

    def testUniform(self):
        self.m.x = gpflow.param.Param(1.0)
        self.m.x.prior = gpflow.priors.Uniform(-2, 3)
        self.m.x.transform = gpflow.transforms.Logistic(-2, 3)

        self.m.set_state(np.random.randn(1))
        p1 = self.m.compute_log_prior()
        self.m.set_state(np.random.randn(1))
        p2 = self.m.compute_log_prior()
        self.assertFalse(p1 == p2)  # prior should no be the same because a transfomration has been applied.


if __name__ == "__main__":
    unittest.main()
