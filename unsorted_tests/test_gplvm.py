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
# limitations under the License.

import tensorflow as tf
import numpy as np

import gpflow
from gpflow.test_util import GPflowTestCase
from gpflow import kernels


np.random.seed(0)

class TestGPLVM(GPflowTestCase):
    def setUp(self):
        # data
        self.N = 20  # number of data points
        D = 5  # data dimension
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn(self.N, D)
        # model
        self.Q = 2  # latent dimensions

    def test_optimise(self):
        with self.test_context():
            m = gpflow.models.GPLVM(self.Y, self.Q)
            linit = m.compute_log_likelihood()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m, maxiter=2)
            self.assertTrue(m.compute_log_likelihood() > linit)

    def test_otherkernel(self):
        with self.test_context():
            k = kernels.Periodic(self.Q)
            XInit = self.rng.rand(self.N, self.Q)
            m = gpflow.models.GPLVM(self.Y, self.Q, XInit, k)
            linit = m.compute_log_likelihood()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m, maxiter=2)
            self.assertTrue(m.compute_log_likelihood() > linit)


class TestBayesianGPLVM(GPflowTestCase):
    def setUp(self):
        # data
        self.N = 20  # number of data points
        self.D = 5  # data dimension
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn(self.N, self.D)
        # model
        self.M = 10  # inducing points

    def test_1d(self):
        with self.test_context():
            Q = 1  # latent dimensions
            k = kernels.RBF(Q)
            Z = np.linspace(0, 1, self.M)
            Z = np.expand_dims(Z, Q)  # inducing points
            m = gpflow.models.BayesianGPLVM(
                X_mean=np.zeros((self.N, Q)),
                X_var=np.ones((self.N, Q)),
                Y=self.Y,
                kern=k,
                M=self.M,
                Z=Z)
            linit = m.compute_log_likelihood()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m, maxiter=2)
            self.assertTrue(m.compute_log_likelihood() > linit)

    def test_2d(self):
        with self.test_context():
            # test default Z on 2_D example
            Q = 2  # latent dimensions
            X_mean = gpflow.models.PCA_reduce(self.Y, Q)
            k = kernels.RBF(Q, ARD=False)
            m = gpflow.models.BayesianGPLVM(
                X_mean=X_mean,
                X_var=np.ones((self.N, Q)),
                Y=self.Y,
                kern=k,
                M=self.M)
            linit = m.compute_log_likelihood()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m, maxiter=2)
            self.assertTrue(m.compute_log_likelihood() > linit)

            # test prediction
            Xtest = self.rng.randn(10, Q)
            mu_f, var_f = m.predict_f(Xtest)
            mu_fFull, var_fFull = m.predict_f_full_cov(Xtest)
            self.assertTrue(np.allclose(mu_fFull, mu_f))
            # check full covariance diagonal
            for i in range(self.D):
                self.assertTrue(np.allclose(var_f[:, i], np.diag(var_fFull[:, :, i])))


if __name__ == "__main__":
    tf.test.main()
