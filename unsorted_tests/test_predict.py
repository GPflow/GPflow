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
# limitations under the License.

import tensorflow as tf
import numpy as np

import gpflow
from gpflow.test_util import GPflowTestCase


class TestGaussian(GPflowTestCase):
    def prepare(self):
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(100, 2)
        self.Y = self.rng.randn(100, 1)
        self.kern = gpflow.kernels.Matern32(2) + gpflow.kernels.White(1)
        self.Xtest = self.rng.randn(10, 2)
        self.Ytest = self.rng.randn(10, 1)
        # make a Gaussian model
        return gpflow.models.GPR(self.X, self.Y, kern=self.kern)

    def test_all(self):
        with self.test_context():
            m = self.prepare()
            mu_f, var_f = m.predict_f(self.Xtest)
            mu_y, var_y = m.predict_y(self.Xtest)

            self.assertTrue(np.allclose(mu_f, mu_y))
            self.assertTrue(np.allclose(var_f, var_y - 1.))

    def test_density(self):
        with self.test_context():
            m = self.prepare()
            mu_y, var_y = m.predict_y(self.Xtest)
            density = m.predict_density(self.Xtest, self.Ytest)

            density_hand = (-0.5 * np.log(2 * np.pi) -
                            0.5 * np.log(var_y) -
                            0.5 * np.square(mu_y - self.Ytest)/var_y)

            self.assertTrue(np.allclose(density_hand, density))

    def test_recompile(self):
        with self.test_context():
            m = self.prepare()
            mu_f, var_f = m.predict_f(self.Xtest)
            mu_y, var_y = m.predict_y(self.Xtest)
            density = m.predict_density(self.Xtest, self.Ytest)

            #change a fix and see if these things still compile
            m.likelihood.variance = 0.2
            m.likelihood.variance.trainable = False

            #this will fail unless a recompile has been triggered
            mu_f, var_f = m.predict_f(self.Xtest)
            mu_y, var_y = m.predict_y(self.Xtest)
            density = m.predict_density(self.Xtest, self.Ytest)


class TestFullCov(GPflowTestCase):
    """
    this base class requires inherriting to specify the model.

    This test structure is more complex that, say, looping over the models, but
    makses all the tests much smaller and so less prone to erroring out. Also,
    if a test fails, it should be clearer where the error is.
    """

    input_dim = 3
    output_dim = 2
    N = 20
    Ntest = 30
    M = 5
    rng = np.random.RandomState(0)
    num_samples = 5
    samples_shape = (num_samples, Ntest, output_dim)
    covar_shape = (output_dim, Ntest, Ntest)
    X = rng.randn(N, input_dim)
    Y = rng.randn(N, output_dim)
    Z = rng.randn(M, input_dim)
    Xtest = rng.randn(Ntest, input_dim)

    @classmethod
    def kernel(cls):
        return gpflow.kernels.Matern32(cls.input_dim)

    def prepare(self):
        return gpflow.models.GPR(self.X, self.Y, kern=self.kernel())

    def test_cov(self):
        with self.test_context():
            m = self.prepare()
            mu1, var = m.predict_f(self.Xtest)
            mu2, covar = m.predict_f_full_cov(self.Xtest)
            self.assertTrue(np.all(mu1 == mu2))
            self.assertTrue(covar.shape == self.covar_shape)
            self.assertTrue(var.shape == (self.Ntest, self.output_dim))
            for i in range(self.output_dim):
                self.assertTrue(np.allclose(var[:, i], np.diag(covar[i, :, :])))

    def test_samples(self):
        with self.test_context():
            m = self.prepare()
            samples = m.predict_f_samples(self.Xtest, self.num_samples)
            print(samples.shape)
            self.assertTrue(samples.shape == self.samples_shape)


class TestFullCovSGPR(TestFullCov):
    def prepare(self):
        return gpflow.models.SGPR(self.X, self.Y, Z=self.Z, kern=self.kernel())


class TestFullCovGPRFITC(TestFullCov):
    def prepare(self):
        return gpflow.models.GPRFITC(self.X, self.Y, Z=self.Z, kern=self.kernel())


class TestFullCovSVGP1(TestFullCov):
    def prepare(self):
        return gpflow.models.SVGP(
            self.X, self.Y, Z=self.Z, kern=self.kernel(),
            likelihood=gpflow.likelihoods.Gaussian(),
            whiten=False, q_diag=True)


class TestFullCovSVGP2(TestFullCov):
    def prepare(self):
        return gpflow.models.SVGP(
            self.X, self.Y, Z=self.Z, kern=self.kernel(),
            likelihood=gpflow.likelihoods.Gaussian(),
            whiten=True, q_diag=False)


class TestFullCovSVGP3(TestFullCov):
    def prepare(self):
        return gpflow.models.SVGP(
            self.X, self.Y, Z=self.Z, kern=self.kernel(),
            likelihood=gpflow.likelihoods.Gaussian(),
            whiten=True, q_diag=True)


class TestFullCovSVGP4(TestFullCov):
    def prepare(self):
        return gpflow.models.SVGP(
            self.X, self.Y, Z=self.Z, kern=self.kernel(),
            likelihood=gpflow.likelihoods.Gaussian(),
            whiten=True, q_diag=False)


class TestFullCovVGP(TestFullCov):
    def prepare(self):
        return gpflow.models.VGP(
            self.X, self.Y, kern=self.kernel(),
            likelihood=gpflow.likelihoods.Gaussian())


class TestFullCovGPMC(TestFullCov):
    def prepare(self):
        return gpflow.models.GPMC(
            self.X, self.Y, kern=self.kernel(),
            likelihood=gpflow.likelihoods.Gaussian())


class TestFullCovSGPMC(TestFullCov):
    def prepare(self):
        return gpflow.models.SGPMC(
            self.X, self.Y, kern=self.kernel(),
            likelihood=gpflow.likelihoods.Gaussian(),
            Z=self.Z)


if __name__ == "__main__":
    tf.test.main()
