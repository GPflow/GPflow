from __future__ import absolute_import, print_function
import gpflow
import tensorflow as tf
import numpy as np
import unittest

from testing.gpflow_testcase import GPflowTestCase
from .reference import referenceRbfKernel, referenceArcCosineKernel, referencePeriodicKernel

class TestRbf(GPflowTestCase):
    def test_1d(self):
        with self.test_session() as sess:
            lengthScale = 1.4
            variance = 2.3
            kernel = gpflow.kernels.RBF(1)
            kernel.lengthscales = lengthScale
            kernel.variance = variance
            rng = np.random.RandomState(1)

            x_free = tf.placeholder('float64')
            kernel.make_tf_array(x_free)
            X = tf.placeholder('float64')
            X_data = rng.randn(3, 1)
            reference_gram_matrix = referenceRbfKernel(X_data, lengthScale, variance)

            with kernel.tf_mode():
                gram_matrix = sess.run(kernel.K(X), feed_dict={x_free: kernel.get_free_state(), X: X_data})
            self.assertTrue(np.allclose(gram_matrix, reference_gram_matrix))


class TestArcCosine(GPflowTestCase):
    def evalKernelError(self, D, variance, weight_variances,
                        bias_variance, order, ARD, X_data):
        with self.test_session() as sess:
            kernel = gpflow.kernels.ArcCosine(
                D,
                order=order,
                variance=variance,
                weight_variances=weight_variances,
                bias_variance=bias_variance,
                ARD=ARD)
            rng = np.random.RandomState(1)

            x_free = tf.placeholder('float64')
            kernel.make_tf_array(x_free)
            X = tf.placeholder('float64')

            if weight_variances is None:
                weight_variances = 1.
            reference_gram_matrix = referenceArcCosineKernel(
                X_data, order,
                weight_variances,
                bias_variance,
                variance)

            with kernel.tf_mode():
                gram_matrix = sess.run(kernel.K(X), feed_dict={x_free: kernel.get_free_state(), X: X_data})

            self.assertTrue(np.allclose(gram_matrix, reference_gram_matrix))

    def test_1d(self):
        with self.test_session():
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
        with self.test_session():
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
        with self.test_session(), self.assertRaises(ValueError):
            gpflow.kernels.ArcCosine(1, order=42)

    def test_weight_initializations(self):
        with self.test_session():
            D = 1
            N = 3
            weight_variances = None
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
        with self.test_session() as sess:
            D = 1
            N = 4

            rng = np.random.RandomState(23)
            X_data = rng.rand(N, D)
            kernel = gpflow.kernels.ArcCosine(D)

            x_free = tf.placeholder('float64')
            kernel.make_tf_array(x_free)
            X = tf.placeholder('float64')

            with kernel.tf_mode():
                gradients = sess.run(tf.gradients(kernel.K(X), X), feed_dict={x_free: kernel.get_free_state(), X: X_data})

            self.assertFalse(np.any(np.isnan(gradients)))


class TestPeriodic(GPflowTestCase):
    def evalKernelError(self, D, lengthscale, variance, period, X_data):
        with self.test_session() as sess:
            kernel = gpflow.kernels.PeriodicKernel(
                D, period=period, variance=variance, lengthscales=lengthscale)

            x_free = tf.placeholder('float64')
            kernel.make_tf_array(x_free)
            X = tf.placeholder('float64')
            reference_gram_matrix = referencePeriodicKernel(
                X_data, lengthscale, variance, period)

            with kernel.tf_mode():
                gram_matrix = sess.run(
                    kernel.K(X), feed_dict={x_free: kernel.get_free_state(), X: X_data})
            self.assertTrue(np.allclose(gram_matrix, reference_gram_matrix))

    def test_1d(self):
        with self.test_session():
            D = 1
            lengthScale = 2
            variance = 2.3
            period = 2
            rng = np.random.RandomState(1)
            X_data = rng.randn(3, 1)
            self.evalKernelError(D, lengthScale, variance, period, X_data)

    def test_2d(self):
        with self.test_session():
            D = 2
            N = 5
            lengthScale = 11.5
            variance = 1.3
            period = 20
            rng = np.random.RandomState(1)
            X_data = rng.multivariate_normal(np.zeros(D), np.eye(D), N)

            self.evalKernelError(D, lengthScale, variance, period, X_data)


class TestCoregion(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.rng = np.random.RandomState(0)
            self.k = gpflow.kernels.Coregion(1, output_dim=3, rank=2)
            self.k.W = self.rng.randn(3, 2)
            self.k.kappa = self.rng.rand(3) + 1.
            self.X = np.random.randint(0, 3, (10, 1))
            self.X2 = np.random.randint(0, 3, (12, 1))

    def test_shape(self):
        with self.test_session():
            K = self.k.compute_K(self.X, self.X2)
            self.assertTrue(K.shape == (10, 12))
            K = self.k.compute_K_symm(self.X)
            self.assertTrue(K.shape == (10, 10))

    def test_diag(self):
        with self.test_session():
            K = self.k.compute_K_symm(self.X)
            Kdiag = self.k.compute_Kdiag(self.X)
            self.assertTrue(np.allclose(np.diag(K), Kdiag))

    def test_slice(self):
        with self.test_session():
            # compute another kernel with additinoal inputs,
            # make sure out kernel is still okay.
            X = np.hstack((self.X, self.rng.randn(10, 1)))
            k1 = gpflow.kernels.Coregion(1, 3, 2, active_dims=[0])
            k2 = gpflow.kernels.RBF(1, active_dims=[1])
            k = k1 * k2
            K1 = k.compute_K_symm(X)
            K2 = k1.compute_K_symm(X) * k2.compute_K_symm(X)  # slicing happens inside kernel
            self.assertTrue(np.allclose(K1, K2))


class TestKernSymmetry(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.kernels = [gpflow.kernels.Constant,
                            gpflow.kernels.Linear,
                            gpflow.kernels.Polynomial,
                            gpflow.kernels.ArcCosine]
            self.kernels += gpflow.kernels.Stationary.__subclasses__()
            self.rng = np.random.RandomState()

    def test_1d(self):
        with self.test_session() as sess:
            kernels = [K(1) for K in self.kernels]
            x_free = tf.placeholder('float64')
            [k.make_tf_array(x_free) for k in kernels]
            X = tf.placeholder('float64')
            X_data = self.rng.randn(10, 1)
            for k in kernels:
                with k.tf_mode():
                    errors = sess.run(
                        k.K(X) - k.K(X, X),
                        feed_dict={x_free: k.get_free_state(), X: X_data})
                    self.assertTrue(np.allclose(errors, 0))

    def test_5d(self):
        with self.test_session() as sess:
            kernels = [K(5) for K in self.kernels]
            x_free = tf.placeholder('float64')
            [k.make_tf_array(x_free) for k in kernels]
            X = tf.placeholder('float64')
            X_data = self.rng.randn(10, 5)
            for k in kernels:
                with k.tf_mode():
                    errors = sess.run(
                        k.K(X) - k.K(X, X),
                        feed_dict={x_free: k.get_free_state(), X: X_data})
                    self.assertTrue(np.allclose(errors, 0))


class TestKernDiags(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            inputdim = 3
            rng = np.random.RandomState(1)
            self.X = tf.placeholder(tf.float64, [30, inputdim])
            self.X_data = rng.randn(30, inputdim)
            self.kernels = [k(inputdim) for k in gpflow.kernels.Stationary.__subclasses__() +
                            [gpflow.kernels.Constant,
                             gpflow.kernels.Linear,
                             gpflow.kernels.Polynomial]]
            self.kernels.append(gpflow.kernels.RBF(inputdim) + gpflow.kernels.Linear(inputdim))
            self.kernels.append(gpflow.kernels.RBF(inputdim) * gpflow.kernels.Linear(inputdim))
            self.kernels.append(gpflow.kernels.RBF(inputdim) +
                                gpflow.kernels.Linear(
                                    inputdim, ARD=True, variance=rng.rand(inputdim)))
            self.kernels.append(gpflow.kernels.PeriodicKernel(inputdim))
            self.kernels.extend(gpflow.kernels.ArcCosine(inputdim, order=order)
                                for order in gpflow.kernels.ArcCosine.implemented_orders)

            self.x_free = tf.placeholder('float64')
            [k.make_tf_array(self.x_free) for k in self.kernels]

    def test(self):
        with self.test_session() as sess:
            for k in self.kernels:
                with k.tf_mode():
                    k1 = k.Kdiag(self.X)
                    k2 = tf.diag_part(k.K(self.X))
                    k1, k2 = sess.run([k1, k2],
                        feed_dict={self.x_free: k.get_free_state(), self.X: self.X_data})
                self.assertTrue(np.allclose(k1, k2))


class TestAdd(GPflowTestCase):
    """
    add a rbf and linear kernel, make sure the result is the same as adding
    the result of the kernels separaetely
    """

    def setUp(self):
        with self.test_session():
            self.rbf = gpflow.kernels.RBF(1)
            self.lin = gpflow.kernels.Linear(1)
            self.k = gpflow.kernels.RBF(1) + gpflow.kernels.Linear(1)
            self.rng = np.random.RandomState(0)

    def test_sym(self):
        with self.test_session() as sess:
            x_free = tf.placeholder('float64')
            [k.make_tf_array(x_free) for k in (self.rbf, self.lin, self.k)]
            X = tf.placeholder('float64')
            X_data = self.rng.randn(10, 1)
            for k in [self.rbf, self.lin, self.k]:
                with k.tf_mode():
                    k._K = sess.run(
                        k.K(X),
                        feed_dict={x_free: k.get_free_state(), X: X_data})
            self.assertTrue(np.allclose(self.rbf._K + self.lin._K, self.k._K))

    def test_asym(self):
        with self.test_session() as sess:
            x_free = tf.placeholder('float64')
            [k.make_tf_array(x_free) for k in (self.rbf, self.lin, self.k)]
            X = tf.placeholder('float64')
            Z = tf.placeholder('float64')
            X_data = self.rng.randn(10, 1)
            Z_data = self.rng.randn(12, 1)
            for k in [self.rbf, self.lin, self.k]:
                with k.tf_mode():
                    k._K = sess.run(
                        k.K(X),
                        feed_dict={x_free: k.get_free_state(), X: X_data, Z: Z_data})
            self.assertTrue(np.allclose(self.rbf._K + self.lin._K, self.k._K))


class TestWhite(GPflowTestCase):
    """
    The white kernel should not give the same result when called with k(X) and
    k(X, X)
    """

    def setUp(self):
        with self.test_session():
            self.k = gpflow.kernels.White(1)
            self.rng = np.random.RandomState(0)

    def test(self):
        with self.test_session() as sess:
            x_free = tf.placeholder('float64')
            self.k.make_tf_array(x_free)
            X = tf.placeholder('float64')
            X_data = self.rng.randn(10, 1)
            with self.k.tf_mode():
                K_sym = sess.run(
                    self.k.K(X),
                    feed_dict={x_free: self.k.get_free_state(), X: X_data})
                K_asym = sess.run(
                    self.k.K(X, X),
                    feed_dict={x_free: self.k.get_free_state(), X: X_data})
            self.assertFalse(np.allclose(K_sym, K_asym))


class TestSlice(GPflowTestCase):
    """
    Make sure the results of a sliced kernel is the same as an unsliced kernel
    with correctly sliced data...
    """

    def setUp(self):
        with self.test_session():
            self.rng = np.random.RandomState(0)

            self.X = self.rng.randn(20, 2)
            self.Z = self.rng.randn(10, 2)

            kernels = [gpflow.kernels.Constant,
                       gpflow.kernels.Linear,
                       gpflow.kernels.Polynomial]
            kernels += gpflow.kernels.Stationary.__subclasses__()
            self.kernels = []
            for kernclass in kernels:
                k1 = kernclass(1, active_dims=[0])
                k2 = kernclass(1, active_dims=[1])
                k3 = kernclass(1, active_dims=slice(0, 1))
                self.kernels.append([k1, k2, k3])

    def test_symm(self):
        with self.test_session():
            for k1, k2, k3 in self.kernels:
                K1 = k1.compute_K_symm(self.X)
                K2 = k2.compute_K_symm(self.X)
                K3 = k3.compute_K_symm(self.X[:, :1])
                K4 = k3.compute_K_symm(self.X[:, 1:])
                self.assertTrue(np.allclose(K1, K3))
                self.assertTrue(np.allclose(K2, K4))

    def test_asymm(self):
        with self.test_session():
            for k1, k2, k3 in self.kernels:
                K1 = k1.compute_K(self.X, self.Z)
                K2 = k2.compute_K(self.X, self.Z)
                K3 = k3.compute_K(self.X[:, :1], self.Z[:, :1])
                K4 = k3.compute_K(self.X[:, 1:], self.Z[:, 1:])
                self.assertTrue(np.allclose(K1, K3))
                self.assertTrue(np.allclose(K2, K4))


class TestProd(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.k1 = gpflow.kernels.Matern32(2)
            self.k2 = gpflow.kernels.Matern52(2, lengthscales=0.3)
            self.k3 = self.k1 * self.k2
            self.x_free = tf.placeholder(tf.float64)
            self.X = tf.placeholder(tf.float64, [30, 2])
            self.X_data = np.random.randn(30, 2)

    def test_prod(self):
        with self.test_session() as sess, self.k1.tf_mode(), self.k2.tf_mode(), self.k3.tf_mode():
            self.k1.make_tf_array(self.x_free)
            K1 = self.k1.K(self.X)
            K1 = sess.run(
                K1, feed_dict={self.X: self.X_data, self.x_free: self.k1.get_free_state()})

            self.k2.make_tf_array(self.x_free)
            K2 = self.k2.K(self.X)
            K2 = sess.run(
                K2, feed_dict={self.X: self.X_data, self.x_free: self.k2.get_free_state()})

            self.k3.make_tf_array(self.x_free)
            K3 = self.k3.K(self.X)
            K3 = sess.run(
                K3, feed_dict={self.X: self.X_data, self.x_free: self.k3.get_free_state()})

            self.assertTrue(np.allclose(K1 * K2, K3))


class TestARDActiveProd(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.rng = np.random.RandomState(0)

            # k3 = k1 * k2
            self.k1 = gpflow.kernels.RBF(3, active_dims=[0, 1, 3], ARD=True)
            self.k2 = gpflow.kernels.RBF(1, active_dims=[2], ARD=True)
            self.k3 = gpflow.kernels.RBF(4, ARD=True)
            self.k1.lengthscales = np.array([3.4, 4.5, 5.6])
            self.k2.lengthscales = 6.7
            self.k3.lengthscales = np.array([3.4, 4.5, 6.7, 5.6])
            self.k3a = self.k1 * self.k2

            # make kernel functions in python
            self.x_free = tf.placeholder('float64')
            self.k3.make_tf_array(self.x_free)
            self.k3a.make_tf_array(self.x_free)
            self.X = tf.placeholder('float64', [50, 4])
            self.X_data = np.random.randn(50, 4)

    def test(self):
        with self.test_session() as sess, self.k3.tf_mode(), self.k3a.tf_mode():
            K1 = self.k3.K(self.X)
            K2 = self.k3a.K(self.X)
            K1 = sess.run(
                K1, feed_dict={self.X: self.X_data, self.x_free: self.k3.get_free_state()})
            K2 = sess.run(
                K2, feed_dict={self.X: self.X_data, self.x_free: self.k3a.get_free_state()})
            self.assertTrue(np.allclose(K1, K2))


class TestKernNaming(GPflowTestCase):
    def test_no_nesting_1(self):
        with self.test_session():
            k1 = gpflow.kernels.RBF(1)
            k2 = gpflow.kernels.Linear(2)
            k3 = k1 + k2
            k4 = gpflow.kernels.Matern32(1)
            k5 = k3 + k4
            self.assertTrue(k5.rbf is k1)
            self.assertTrue(k5.linear is k2)
            self.assertTrue(k5.matern32 is k4)

    def test_no_nesting_2(self):
        with self.test_session():
            k1 = gpflow.kernels.RBF(1) + gpflow.kernels.Linear(2)
            k2 = gpflow.kernels.Matern32(1) + gpflow.kernels.Matern52(2)
            k = k1 + k2
            self.assertTrue(hasattr(k, 'rbf'))
            self.assertTrue(hasattr(k, 'linear'))
            self.assertTrue(hasattr(k, 'matern32'))
            self.assertTrue(hasattr(k, 'matern52'))

    def test_simple(self):
        with self.test_session():
            k1 = gpflow.kernels.RBF(1)
            k2 = gpflow.kernels.Linear(2)
            k = k1 + k2
            self.assertTrue(k.rbf is k1)
            self.assertTrue(k.linear is k2)

    def test_duplicates_1(self):
        with self.test_session():
            k1 = gpflow.kernels.Matern32(1)
            k2 = gpflow.kernels.Matern32(43)
            k = k1 + k2
            self.assertTrue(k.matern32_1 is k1)
            self.assertTrue(k.matern32_2 is k2)

    def test_duplicates_2(self):
        with self.test_session():
            k1 = gpflow.kernels.Matern32(1)
            k2 = gpflow.kernels.Matern32(2)
            k3 = gpflow.kernels.Matern32(3)
            k = k1 + k2 + k3
            self.assertTrue(k.matern32_1 is k1)
            self.assertTrue(k.matern32_2 is k2)
            self.assertTrue(k.matern32_3 is k3)


class TestKernNamingProduct(GPflowTestCase):
    def test_no_nesting_1(self):
        with self.test_session():
            k1 = gpflow.kernels.RBF(1)
            k2 = gpflow.kernels.Linear(2)
            k3 = k1 * k2
            k4 = gpflow.kernels.Matern32(1)
            k5 = k3 * k4
            self.assertTrue(k5.rbf is k1)
            self.assertTrue(k5.linear is k2)
            self.assertTrue(k5.matern32 is k4)

    def test_no_nesting_2(self):
        with self.test_session():
            k1 = gpflow.kernels.RBF(1) * gpflow.kernels.Linear(2)
            k2 = gpflow.kernels.Matern32(1) * gpflow.kernels.Matern52(2)
            k = k1 * k2
            self.assertTrue(hasattr(k, 'rbf'))
            self.assertTrue(hasattr(k, 'linear'))
            self.assertTrue(hasattr(k, 'matern32'))
            self.assertTrue(hasattr(k, 'matern52'))

    def test_simple(self):
        with self.test_session():
            k1 = gpflow.kernels.RBF(1)
            k2 = gpflow.kernels.Linear(2)
            k = k1 * k2
            self.assertTrue(k.rbf is k1)
            self.assertTrue(k.linear is k2)

    def test_duplicates_1(self):
        with self.test_session():
            k1 = gpflow.kernels.Matern32(1)
            k2 = gpflow.kernels.Matern32(43)
            k = k1 * k2
            self.assertTrue(k.matern32_1 is k1)
            self.assertTrue(k.matern32_2 is k2)

    def test_duplicates_2(self):
        with self.test_session():
            k1 = gpflow.kernels.Matern32(1)
            k2 = gpflow.kernels.Matern32(2)
            k3 = gpflow.kernels.Matern32(3)
            k = k1 * k2 * k3
            self.assertTrue(k.matern32_1 is k1)
            self.assertTrue(k.matern32_2 is k2)
            self.assertTrue(k.matern32_3 is k3)


class TestARDInit(GPflowTestCase):
    """
    For ARD kernels, make sure that kernels can be instantiated with a single
    lengthscale or a suitable array of lengthscales
    """

    def test_scalar(self):
        with self.test_session():
            k1 = gpflow.kernels.RBF(3, lengthscales=2.3)
            k2 = gpflow.kernels.RBF(3, lengthscales=np.ones(3) * 2.3)
            self.assertTrue(np.all(k1.lengthscales.value == k2.lengthscales.value))

    def test_MLP(self):
        with self.test_session():
            k1 = gpflow.kernels.ArcCosine(3, weight_variances=1.23, ARD=True)
            k2 = gpflow.kernels.ArcCosine(3, weight_variances=np.ones(3) * 1.23, ARD=True)
            self.assertTrue(np.all(k1.weight_variances.value == k2.weight_variances.value))

if __name__ == "__main__":
    unittest.main()
