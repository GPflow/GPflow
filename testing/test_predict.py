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

import gpflow
import numpy as np
import unittest
import tensorflow as tf

from testing.gpflow_testcase import GPflowTestCase


class TestGaussian(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.rng = np.random.RandomState(0)
            self.X = self.rng.randn(100,2)
            self.Y = self.rng.randn(100, 1)
            self.kern = gpflow.kernels.Matern32(2) + gpflow.kernels.White(1)
            self.Xtest = self.rng.randn(10, 2)
            self.Ytest = self.rng.randn(10, 1)
            # make a Gaussian model
            self.m = gpflow.gpr.GPR(self.X, self.Y, kern=self.kern)

    def test_all(self):
        with self.test_session():
            mu_f, var_f = self.m.predict_f(self.Xtest)
            mu_y, var_y = self.m.predict_y(self.Xtest)

            self.assertTrue(np.allclose(mu_f, mu_y))
            self.assertTrue(np.allclose(var_f, var_y - 1.))

    def test_density(self):
        with self.test_session():
            mu_y, var_y = self.m.predict_y(self.Xtest)
            density = self.m.predict_density(self.Xtest, self.Ytest)

            density_hand = -0.5*np.log(2*np.pi) - 0.5*np.log(var_y) - 0.5*np.square(mu_y - self.Ytest)/var_y
            self.assertTrue(np.allclose(density_hand, density))

    def test_recompile(self):
        with self.test_session():
            mu_f, var_f = self.m.predict_f(self.Xtest)
            mu_y, var_y = self.m.predict_y(self.Xtest)
            density = self.m.predict_density(self.Xtest, self.Ytest)

            #change a fix and see if these things still compile
            self.m.likelihood.variance = 0.2
            self.m.likelihood.variance.fixed = True

            #this will fail unless a recompile has been triggered
            mu_f, var_f = self.m.predict_f(self.Xtest)
            mu_y, var_y = self.m.predict_y(self.Xtest)
            density = self.m.predict_density(self.Xtest, self.Ytest)


class TestFullCov(GPflowTestCase):
    """
    this base class requires inherriting to specify the model.

    This test structure is more complex that, say, looping over the models, but
    makses all the tests much smaller and so less prone to erroring out. Also,
    if a test fails, it should be clearer where the error is.
    """
    def setUp(self):
        with self.test_session():
            self.input_dim = 3
            self.output_dim = 2
            self.N = 20
            self.Ntest = 30
            self.M = 5
            rng = np.random.RandomState(0)
            self.num_samples = 5
            self.samples_shape = (self.num_samples, self.Ntest, self.output_dim)
            self.covar_shape = (self.Ntest, self.Ntest, self.output_dim)
            self.X, self.Y, self.Z, self.Xtest = (
                rng.randn(self.N, self.input_dim),
                rng.randn(self.N, self.output_dim),
                rng.randn(self.M, self.input_dim),
                rng.randn(self.Ntest, self.input_dim))
            self.k = lambda: gpflow.kernels.Matern32(self.input_dim)
            self.model = gpflow.gpr.GPR(self.X, self.Y, kern=self.k())

    def test_cov(self):
        with self.test_session():
            mu1, var = self.model.predict_f(self.Xtest)
            mu2, covar = self.model.predict_f_full_cov(self.Xtest)
            self.assertTrue(np.all(mu1 == mu2))
            self.assertTrue(covar.shape == self.covar_shape)
            self.assertTrue(var.shape == (self.Ntest, self.output_dim))
            for i in range(self.output_dim):
                self.assertTrue(np.allclose(var[:, i], np.diag(covar[:, :, i])))

    def test_samples(self):
        with self.test_session():
            samples = self.model.predict_f_samples(self.Xtest, self.num_samples)
            self.assertTrue(samples.shape == self.samples_shape)


class TestFullCovSGPR(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.sgpr.SGPR(self.X, self.Y, Z=self.Z, kern=self.k())


class TestFullCovGPRFITC(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.sgpr.GPRFITC(
                self.X, self.Y,
                Z=self.Z, kern=self.k())


class TestFullCovSVGP1(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.svgp.SVGP(
                self.X, self.Y, Z=self.Z, kern=self.k(),
                likelihood=gpflow.likelihoods.Gaussian(),
                whiten=False, q_diag=True)


class TestFullCovSVGP2(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.svgp.SVGP(
                self.X, self.Y, Z=self.Z, kern=self.k(),
                likelihood=gpflow.likelihoods.Gaussian(),
                whiten=True, q_diag=False)


class TestFullCovSVGP3(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.svgp.SVGP(
                self.X, self.Y, Z=self.Z, kern=self.k(),
                likelihood=gpflow.likelihoods.Gaussian(),
                whiten=True, q_diag=True)


class TestFullCovSVGP4(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.svgp.SVGP(
                self.X, self.Y, Z=self.Z, kern=self.k(),
                likelihood=gpflow.likelihoods.Gaussian(),
                whiten=True, q_diag=False)


class TestFullCovVGP(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.vgp.VGP(
                self.X, self.Y, kern=self.k(),
                likelihood=gpflow.likelihoods.Gaussian())


class TestFullCovGPMC(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.gpmc.GPMC(
                self.X, self.Y, kern=self.k(),
                likelihood=gpflow.likelihoods.Gaussian())


class TestFullCovSGPMC(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        with self.test_session():
            self.model = gpflow.sgpmc.SGPMC(
                self.X, self.Y, kern=self.k(),
                likelihood=gpflow.likelihoods.Gaussian(),
                Z=self.Z)


if __name__ == "__main__":
    unittest.main()
