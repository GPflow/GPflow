# Copyright 2017 Mark van der Wilk
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
import tensorflow as tf

import gpflow
from gpflow.test_util import GPflowTestCase


class TestInducingPoints(GPflowTestCase):
    def test_feature_len(self):
        N, D = 17, 3
        Z = np.random.randn(N, D)
        f = gpflow.features.InducingPoints(Z)

        self.assertTrue(len(f), N)
        with gpflow.decors.params_as_tensors_for(f):
            self.assertTrue(len(f), N)
        # GPflow does not support re-assignment with different shapes at the moment

    def test_inducing_points_equivalence(self):
        # Inducing features must be the same as the kernel evaluations
        with self.test_context() as session:
            Z = np.random.randn(101, 3)
            f = gpflow.features.InducingPoints(Z)

            kernels = [gpflow.kernels.RBF(3, 0.46, lengthscales=np.array([0.143, 1.84, 2.0]), ARD=True),
                       gpflow.kernels.PeriodicKernel(3, 0.4, 1.8)]

            for k in kernels:
                self.assertTrue(np.allclose(session.run(f.Kuu(k)), k.compute_K_symm(Z)))


class TestMultiScaleInducing(GPflowTestCase):
    def prepare(self):
        rbf = gpflow.kernels.RBF(2, 1.3441, lengthscales=np.array([0.3414, 1.234]))
        Z = np.random.randn(23, 3)
        scales = 1.0 + np.random.rand(*Z.shape) * 0.3
        feature = gpflow.features.Multiscale(Z, scales)
        feature_0lengthscale = gpflow.features.Multiscale(Z, np.zeros(Z.shape))
        feature_inducingpoint = gpflow.features.InducingPoints(Z)
        return [rbf, feature, feature_0lengthscale, feature_inducingpoint]

    def test_equivalence_inducing_points(self):
        # Multiscale must be equivalent to inducing points when variance is zero
        with self.test_context() as session:
            rbf, feature, feature_0lengthscale, feature_inducingpoint = self.prepare()
            Xnew = np.random.randn(13, 3)

            ms, point = session.run([feature_0lengthscale.Kuf(rbf, Xnew), feature_inducingpoint.Kuf(rbf, Xnew)])
            pd = np.max(np.abs(ms - point) / point * 100)
            self.assertTrue(pd < 0.1)

            ms, point = session.run([feature_0lengthscale.Kuu(rbf), feature_inducingpoint.Kuu(rbf)])
            pd = np.max(np.abs(ms - point) / point * 100)
            self.assertTrue(pd < 0.1)

    def test_matrix_psd(self):
        # Conditional variance must be PSD.
        Xnew = np.random.randn(13, 3)
        with self.test_context() as session:
            rbf, feature, feature_0lengthscale, feature_inducingpoint = self.prepare()
            Kuf, Kuu = session.run([feature.Kuf(rbf, Xnew), feature.Kuu(rbf, jitter=gpflow.settings.jitter)])
            Kff = rbf.compute_K_symm(Xnew)
        Qff = Kuf.T @ np.linalg.solve(Kuu, Kuf)
        self.assertTrue(np.all(np.linalg.eig(Kff - Qff)[0] > 0.0))


if __name__ == "__main__":
    tf.test.main()
