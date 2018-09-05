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
from numpy.testing import assert_allclose

import copy
import gpflow
from gpflow.test_util import GPflowTestCase, session_tf

from .reference import referenceRbfKernel, referenceArcCosineKernel, referencePeriodicKernel

class TestRbf(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()

    def test_1d(self):
        with self.test_context() as session:
            lengthscale = 1.4
            variance = 2.3
            kernel = gpflow.kernels.RBF(1, lengthscales=lengthscale, variance=variance)
            rng = np.random.RandomState(1)

            X = tf.placeholder(gpflow.settings.float_type)
            X_data = rng.randn(3, 1).astype(gpflow.settings.float_type)

            kernel.compile()
            gram_matrix = session.run(kernel.K(X), feed_dict={X: X_data})
            reference_gram_matrix = referenceRbfKernel(X_data, lengthscale, variance)
            self.assertTrue(np.allclose(gram_matrix, reference_gram_matrix))


class TestRQ(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()

    def test_1d(self):
        with self.test_context() as session:
            lengthscale = 1.4
            variance = 2.3
            kSE = gpflow.kernels.RBF(1, lengthscales=lengthscale, variance=variance)
            kRQ = gpflow.kernels.RationalQuadratic(1, lengthscales=lengthscale, variance=variance, alpha=1e8)
            rng = np.random.RandomState(1)

            X = tf.placeholder(gpflow.settings.float_type)
            X_data = rng.randn(6, 1).astype(gpflow.settings.float_type)

            kSE.compile()
            kRQ.compile()
            gram_matrix_SE = session.run(kSE.K(X), feed_dict={X: X_data})
            gram_matrix_RQ = session.run(kRQ.K(X), feed_dict={X: X_data})
            np.testing.assert_allclose(gram_matrix_SE, gram_matrix_RQ)


class TestArcCosine(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()

    def evalKernelError(self, D, variance, weight_variances,
                        bias_variance, order, ARD, X_data):
        with self.test_context() as session:
            kernel = gpflow.kernels.ArcCosine(
                D,
                order=order,
                variance=variance,
                weight_variances=weight_variances,
                bias_variance=bias_variance,
                ARD=ARD)

            if weight_variances is None:
                weight_variances = 1.
            kernel.compile()
            X = tf.placeholder(gpflow.settings.float_type)
            gram_matrix = session.run(kernel.K(X), feed_dict={X: X_data})
            reference_gram_matrix = referenceArcCosineKernel(
                X_data, order,
                weight_variances,
                bias_variance,
                variance)

            assert_allclose(gram_matrix, reference_gram_matrix)

    def test_1d(self):
        with self.test_context():
            D = 1
            N = 3
            weight_variances = 1.7
            bias_variance = 0.6
            variance = 2.3
            ARD = False
            orders = gpflow.kernels.ArcCosine.implemented_orders

            rng = np.random.RandomState(1)
            X_data = rng.randn(N, D)
            for order in orders:
                self.evalKernelError(D, variance, weight_variances,
                                     bias_variance, order, ARD, X_data)

    def test_3d(self):
        with self.test_context():
            D = 3
            N = 8
            weight_variances = np.array([0.4, 4.2, 2.3])
            bias_variance = 1.9
            variance = 1e-2
            ARD = True
            orders = gpflow.kernels.ArcCosine.implemented_orders

            rng = np.random.RandomState(1)
            X_data = rng.randn(N, D)
            for order in orders:
                self.evalKernelError(D, variance, weight_variances,
                                     bias_variance, order, ARD, X_data)

    def test_non_implemented_order(self):
        with self.test_context():
            with self.assertRaises(ValueError):
                gpflow.kernels.ArcCosine(1, order=42)

    def test_weight_initializations(self):
        with self.test_context():
            D = 1
            N = 3
            weight_variances = 1.
            bias_variance = 1.
            variance = 1.
            ARDs = {False, True}
            order = 0

            rng = np.random.RandomState(1)
            X_data = rng.randn(N, D)
            for ARD in ARDs:
                self.evalKernelError(
                    D, variance, weight_variances,
                    bias_variance, order, ARD, X_data)

    def test_nan_in_gradient(self):
        with self.test_context() as session:
            D = 1
            N = 4

            rng = np.random.RandomState(23)
            X_data = rng.rand(N, D)
            kernel = gpflow.kernels.ArcCosine(D)

            X = tf.placeholder(tf.float64)
            kernel.compile()
            grads = tf.gradients(kernel.K(X), X)
            gradients = session.run(grads, feed_dict={X: X_data})
            self.assertFalse(np.any(np.isnan(gradients)))


class TestPeriodic(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()

    def evalKernelError(self, D, lengthscale, variance, period, X_data):
        with self.test_context() as session:
            kernel = gpflow.kernels.Periodic(
                D, period=period, variance=variance, lengthscales=lengthscale)

            X = tf.placeholder(gpflow.settings.float_type)
            reference_gram_matrix = referencePeriodicKernel(
                X_data, lengthscale, variance, period)
            kernel.compile()
            gram_matrix = session.run(kernel.K(X), feed_dict={X: X_data})
            assert_allclose(gram_matrix, reference_gram_matrix)

    def test_1d(self):
        with self.test_context():
            D = 1
            lengthScale = 2.0
            variance = 2.3
            period = 2.
            rng = np.random.RandomState(1)
            X_data = rng.randn(3, 1)
            self.evalKernelError(D, lengthScale, variance, period, X_data)

    def test_2d(self):
        with self.test_context():
            D = 2
            N = 5
            lengthScale = 11.5
            variance = 1.3
            period = 20.
            rng = np.random.RandomState(1)
            X_data = rng.multivariate_normal(np.zeros(D), np.eye(D), N)
            self.evalKernelError(D, lengthScale, variance, period, X_data)


class TestCoregion(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            self.rng = np.random.RandomState(0)
            self.k = gpflow.kernels.Coregion(1, output_dim=3, rank=2)
            self.k.W = self.rng.randn(3, 2)
            self.k.kappa = self.rng.rand(3) + 1.
            self.X = np.random.randint(0, 3, (10, 1))
            self.X2 = np.random.randint(0, 3, (12, 1))

    def tearDown(self):
        GPflowTestCase.tearDown(self)
        self.k.clear()

    def test_shape(self):
        with self.test_context():
            self.k.compile()
            K = self.k.compute_K(self.X, self.X2)
            self.assertTrue(K.shape == (10, 12))
            K = self.k.compute_K_symm(self.X)
            self.assertTrue(K.shape == (10, 10))

    def test_diag(self):
        with self.test_context():
            self.k.compile()
            K = self.k.compute_K_symm(self.X)
            Kdiag = self.k.compute_Kdiag(self.X)
            self.assertTrue(np.allclose(np.diag(K), Kdiag))

    def test_slice(self):
        with self.test_context():
            # compute another kernel with additinoal inputs,
            # make sure out kernel is still okay.
            X = np.hstack((self.X, self.rng.randn(10, 1)))
            k1 = gpflow.kernels.Coregion(1, 3, 2, active_dims=[0])
            k2 = gpflow.kernels.RBF(1, active_dims=[1])
            k = k1 * k2
            k.compile()
            K1 = k.compute_K_symm(X)
            K2 = k1.compute_K_symm(X) * k2.compute_K_symm(X)  # slicing happens inside kernel
            self.assertTrue(np.allclose(K1, K2))


class TestKernSymmetry(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        self.kernels = [gpflow.kernels.Constant,
                        gpflow.kernels.Linear,
                        gpflow.kernels.Polynomial,
                        gpflow.kernels.ArcCosine]
        self.kernels += gpflow.kernels.Stationary.__subclasses__()
        self.rng = np.random.RandomState()

    def test_1d(self):
        with self.test_context() as session:
            kernels = [K(1) for K in self.kernels]
            for kernel in kernels:
                kernel.compile()
            X = tf.placeholder(tf.float64)
            X_data = self.rng.randn(10, 1)
            for k in kernels:
                errors = session.run(k.K(X) - k.K(X, X), feed_dict={X: X_data})
                self.assertTrue(np.allclose(errors, 0))

    def test_5d(self):
        with self.test_context() as session:
            kernels = [K(5) for K in self.kernels]
            for kernel in kernels:
                kernel.compile()
            X = tf.placeholder(tf.float64)
            X_data = self.rng.randn(10, 5)
            for k in kernels:
                errors = session.run(k.K(X) - k.K(X, X), feed_dict={X: X_data})
                self.assertTrue(np.allclose(errors, 0))


class TestKernDiags(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            inputdim = 3
            rng = np.random.RandomState(1)
            self.rng = rng
            self.dim = inputdim
            self.kernels = [k(inputdim) for k in gpflow.kernels.Stationary.__subclasses__() +
                            [gpflow.kernels.Constant,
                             gpflow.kernels.Linear,
                             gpflow.kernels.Polynomial]]
            self.kernels.append(gpflow.kernels.RBF(inputdim) + gpflow.kernels.Linear(inputdim))
            self.kernels.append(gpflow.kernels.RBF(inputdim) * gpflow.kernels.Linear(inputdim))
            self.kernels.append(gpflow.kernels.RBF(inputdim) +
                                gpflow.kernels.Linear(
                                    inputdim, ARD=True, variance=rng.rand(inputdim)))
            self.kernels.append(gpflow.kernels.Periodic(inputdim))
            self.kernels.extend(gpflow.kernels.ArcCosine(inputdim, order=order)
                                for order in gpflow.kernels.ArcCosine.implemented_orders)

    def test(self):
        with self.test_context() as session:
            for k in self.kernels:
                k.initialize(session=session, force=True)
                X = tf.placeholder(tf.float64, [30, self.dim])
                rng = np.random.RandomState(1)
                X_data = rng.randn(30, self.dim)
                k1 = k.Kdiag(X)
                k2 = tf.diag_part(k.K(X))
                k1, k2 = session.run([k1, k2], feed_dict={X: X_data})
                self.assertTrue(np.allclose(k1, k2))


class TestAdd(GPflowTestCase):
    """
    add a rbf and linear kernel, make sure the result is the same as adding
    the result of the kernels separaetely
    """

    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            rbf = gpflow.kernels.RBF(1)
            lin = gpflow.kernels.Linear(1)
            k = (gpflow.kernels.RBF(1, name='RBFInAdd') +
                 gpflow.kernels.Linear(1, name='LinearInAdd'))
            self.rng = np.random.RandomState(0)
            self.kernels = [rbf, lin, k]

    def test_sym(self):
        with self.test_context() as session:
            X = tf.placeholder(tf.float64)
            X_data = self.rng.randn(10, 1)
            res = []
            for k in self.kernels:
                k.compile()
                res.append(session.run(k.K(X), feed_dict={X: X_data}))
            self.assertTrue(np.allclose(res[0] + res[1], res[2]))

    def test_asym(self):
        with self.test_context() as session:
            X = tf.placeholder(tf.float64)
            Z = tf.placeholder(tf.float64)
            X_data = self.rng.randn(10, 1)
            Z_data = self.rng.randn(12, 1)
            res = []
            for k in self.kernels:
                k.compile()
                res.append(session.run(k.K(X, Z), feed_dict={X: X_data, Z: Z_data}))
            self.assertTrue(np.allclose(res[0] + res[1], res[2]))


class TestWhite(GPflowTestCase):
    """
    The white kernel should not give the same result when called with k(X) and
    k(X, X)
    """

    def test(self):
        with self.test_context() as session:
            rng = np.random.RandomState(0)
            X = tf.placeholder(tf.float64)
            X_data = rng.randn(10, 1)
            k = gpflow.kernels.White(1)
            k.compile()
            K_sym = session.run(k.K(X), feed_dict={X: X_data})
            K_asym = session.run(k.K(X, X), feed_dict={X: X_data})
            self.assertFalse(np.allclose(K_sym, K_asym))


class TestSlice(GPflowTestCase):
    """
    Make sure the results of a sliced kernel is the same as an unsliced kernel
    with correctly sliced data...
    """

    def kernels(self):
        ks = [gpflow.kernels.Constant,
              gpflow.kernels.Linear,
              gpflow.kernels.Polynomial]
        ks += gpflow.kernels.Stationary.__subclasses__()
        kernels = []
        kernname = lambda cls, index: '_'.join([cls.__name__, str(index)])
        for kernclass in ks:
            kern = copy.deepcopy(kernclass)
            k1 = lambda: kern(1, active_dims=[0], name=kernname(kern, 1))
            k2 = lambda: kern(1, active_dims=[1], name=kernname(kern, 2))
            k3 = lambda: kern(1, active_dims=slice(0, 1), name=kernname(kern, 3))
            kernels.append([k1, k2, k3])
        return kernels

    def test_symm(self):
        for k1, k2, k3 in self.kernels():
            with self.test_context(graph=tf.Graph()):
                rng = np.random.RandomState(0)
                X = rng.randn(20, 2)
                k1i, k2i, k3i = k1(), k2(), k3()
                K1 = k1i.compute_K_symm(X)
                K2 = k2i.compute_K_symm(X)
                K3 = k3i.compute_K_symm(X[:, :1])
                K4 = k3i.compute_K_symm(X[:, 1:])
                self.assertTrue(np.allclose(K1, K3))
                self.assertTrue(np.allclose(K2, K4))

    def test_asymm(self):
        for k1, k2, k3 in self.kernels():
            with self.test_context(graph=tf.Graph()):
                rng = np.random.RandomState(0)
                X = rng.randn(20, 2)
                Z = rng.randn(10, 2)
                k1i, k2i, k3i = k1(), k2(), k3()
                K1 = k1i.compute_K(X, Z)
                K2 = k2i.compute_K(X, Z)
                K3 = k3i.compute_K(X[:, :1], Z[:, :1])
                K4 = k3i.compute_K(X[:, 1:], Z[:, 1:])
                self.assertTrue(np.allclose(K1, K3))
                self.assertTrue(np.allclose(K2, K4))


class TestProd(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            k1 = gpflow.kernels.Matern32(2)
            k2 = gpflow.kernels.Matern52(2, lengthscales=0.3)
            k3 = k1 * k2
            self.kernels = [k1, k2, k3]

    def tearDown(self):
        GPflowTestCase.tearDown(self)
        self.kernels[2].clear()

    def test_prod(self):
        with self.test_context() as session:
            self.kernels[2].compile()

            X = tf.placeholder(tf.float64, [30, 2])
            X_data = np.random.randn(30, 2)

            res = []
            for kernel in self.kernels:
                K = kernel.K(X)
                res.append(session.run(K, feed_dict={X: X_data}))

            self.assertTrue(np.allclose(res[0] * res[1], res[2]))


class TestARDActiveProd(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        self.rng = np.random.RandomState(0)
        with self.test_context():
            # k3 = k1 * k2
            self.k1 = gpflow.kernels.RBF(3, active_dims=[0, 1, 3], ARD=True)
            self.k2 = gpflow.kernels.RBF(1, active_dims=[2], ARD=True)
            self.k3 = gpflow.kernels.RBF(4, ARD=True)
            self.k1.lengthscales = np.array([3.4, 4.5, 5.6])
            self.k2.lengthscales = np.array([6.7])
            self.k3.lengthscales = np.array([3.4, 4.5, 6.7, 5.6])
            self.k3a = self.k1 * self.k2

    def test(self):
        with self.test_context() as session:
            X = tf.placeholder(tf.float64, [50, 4])
            X_data = np.random.randn(50, 4)
            self.k3.compile()
            self.k3a.compile()
            K1 = self.k3.K(X)
            K2 = self.k3a.K(X)
            K1 = session.run(K1, feed_dict={X: X_data})
            K2 = session.run(K2, feed_dict={X: X_data})
            self.assertTrue(np.allclose(K1, K2))


class TestARDInit(GPflowTestCase):
    """
    For ARD kernels, make sure that kernels can be instantiated with a single
    lengthscale or a suitable array of lengthscales
    """

    def setUp(self):
        self.test_graph = tf.Graph()

    def test_scalar(self):
        with self.test_context():
            k1 = gpflow.kernels.RBF(3, lengthscales=2.3)
            k2 = gpflow.kernels.RBF(3, lengthscales=np.ones(3) * 2.3, ARD=True)
            k1_lengthscales = k1.lengthscales.read_value()
            k2_lengthscales = k2.lengthscales.read_value()
            self.assertTrue(np.all(k1_lengthscales == k2_lengthscales))

    def test_init(self):
        for ARD in (False, True, None):
            with self.assertRaises(ValueError):
                k1 = gpflow.kernels.RBF(1, lengthscales=[1., 1.], ARD=ARD)
            with self.assertRaises(ValueError):
                k2 = gpflow.kernels.RBF(2, lengthscales=[1., 1., 1.], ARD=ARD)

    def test_MLP(self):
        with self.test_context():
            k1 = gpflow.kernels.ArcCosine(3, weight_variances=1.23, ARD=True)
            k2 = gpflow.kernels.ArcCosine(3, weight_variances=np.ones(3) * 1.23, ARD=True)
            k1_variances = k1.weight_variances.read_value()
            k2_variances = k2.weight_variances.read_value()
            self.assertTrue(np.all(k1_variances == k2_variances))

def test_slice_active_dim_regression(session_tf):
    """ Check that we can instantiate a kernel with active_dims given as a slice object """
    gpflow.kernels.RBF(2, active_dims=slice(1, 3, 1))

if __name__ == "__main__":
    tf.test.main()
