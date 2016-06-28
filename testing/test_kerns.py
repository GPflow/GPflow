import GPflow
import tensorflow as tf
import numpy as np
import unittest
from reference import referenceRbfKernel, referencePeriodicKernel


class TestRbf(unittest.TestCase):
    def test_1d(self):
        lengthScale = 1.4
        variance = 2.3
        kernel = GPflow.kernels.RBF(1)
        kernel.lengthscales = lengthScale
        kernel.variance = variance
        rng = np.random.RandomState(1)

        x_free = tf.placeholder('float64')
        kernel.make_tf_array(x_free)
        X = tf.placeholder('float64')
        X_data = rng.randn(3, 1)
        reference_gram_matrix = referenceRbfKernel(X_data, lengthScale, variance)

        with kernel.tf_mode():
            gram_matrix = tf.Session().run(kernel.K(X), feed_dict={x_free: kernel.get_free_state(), X: X_data})
        self.failUnless(np.allclose(gram_matrix-reference_gram_matrix, 0))


class TestPeriodic(unittest.TestCase):
    def evalKernelError(self, D, lengthscale, variance, period, X_data):
        kernel = GPflow.kernels.PeriodicKernel(D, period=period, variance=variance, lengthscales=lengthscale)

        x_free = tf.placeholder('float64')
        kernel.make_tf_array(x_free)
        X = tf.placeholder('float64')
        reference_gram_matrix = referencePeriodicKernel(X_data, lengthscale, variance, period)

        with kernel.tf_mode():
            gram_matrix = tf.Session().run(kernel.K(X),
                                           feed_dict={x_free: kernel.get_free_state(), X: X_data})
        self.failUnless(np.allclose(gram_matrix-reference_gram_matrix, 0))

    def test_1d(self):
        D = 1
        lengthScale = 2
        variance = 2.3
        period = 2
        rng = np.random.RandomState(1)
        X_data = rng.randn(3, 1)
        self.evalKernelError(D, lengthScale, variance, period, X_data)

    def test_2d(self):
        D = 2
        N = 5
        lengthScale = 11.5
        variance = 1.3
        period = 20
        rng = np.random.RandomState(1)
        X_data = rng.multivariate_normal(np.zeros(D), np.eye(D), N)

        self.evalKernelError(D, lengthScale, variance, period, X_data)


class TestKernSymmetry(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.kernels = GPflow.kernels.Stationary.__subclasses__() + [GPflow.kernels.Constant, GPflow.kernels.Linear]
        self.rng = np.random.RandomState()

    def test_1d(self):
        kernels = [K(1) for K in self.kernels]
        x_free = tf.placeholder('float64')
        [k.make_tf_array(x_free) for k in kernels]
        X = tf.placeholder('float64')
        X_data = self.rng.randn(10, 1)
        for k in kernels:
            with k.tf_mode():
                Errors = tf.Session().run(k.K(X) - k.K(X, X),
                                          feed_dict={x_free: k.get_free_state(), X: X_data})
                self.failUnless(np.allclose(Errors, 0))

    def test_5d(self):
        kernels = [K(5) for K in self.kernels]
        x_free = tf.placeholder('float64')
        [k.make_tf_array(x_free) for k in kernels]
        X = tf.placeholder('float64')
        X_data = self.rng.randn(10, 5)
        for k in kernels:
            with k.tf_mode():
                Errors = tf.Session().run(k.K(X) - k.K(X, X),
                                          feed_dict={x_free: k.get_free_state(), X: X_data})
                self.failUnless(np.allclose(Errors, 0))


class TestKernDiags(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        inputdim = 3
        rng = np.random.RandomState(1)
        self.X = tf.placeholder(tf.float64, [30, inputdim])
        self.X_data = rng.randn(30, inputdim)
        self.kernels = [k(inputdim) for k in GPflow.kernels.Stationary.__subclasses__() +
                        [GPflow.kernels.Constant, GPflow.kernels.Linear]]
        self.kernels.append(GPflow.kernels.RBF(inputdim) + GPflow.kernels.Linear(inputdim))
        self.kernels.append(GPflow.kernels.RBF(inputdim) * GPflow.kernels.Linear(inputdim))
        self.kernels.append(GPflow.kernels.RBF(inputdim) +
                            GPflow.kernels.Linear(inputdim, ARD=True, variance=rng.rand(inputdim)))
        self.kernels.append(GPflow.kernels.PeriodicKernel(inputdim))

        self.x_free = tf.placeholder('float64')
        [k.make_tf_array(self.x_free) for k in self.kernels]

    def test(self):
        for k in self.kernels:
            with k.tf_mode():
                k1 = k.Kdiag(self.X)
                k2 = tf.diag_part(k.K(self.X))
                k1, k2 = tf.Session().run([k1, k2],
                                          feed_dict={self.x_free: k.get_free_state(), self.X: self.X_data})
            self.failUnless(np.allclose(k1, k2))


class TestAdd(unittest.TestCase):
    """
    add a rbf and linear kernel, make sure the result is the same as adding
    the result of the kernels separaetely
    """
    def setUp(self):
        tf.reset_default_graph()
        self.rbf = GPflow.kernels.RBF(1)
        self.lin = GPflow.kernels.Linear(1)
        self.k = GPflow.kernels.RBF(1) + GPflow.kernels.Linear(1)
        self.rng = np.random.RandomState(0)

    def test_sym(self):
        x_free = tf.placeholder('float64')
        [k.make_tf_array(x_free) for k in (self.rbf, self.lin, self.k)]
        X = tf.placeholder('float64')
        X_data = self.rng.randn(10, 1)
        for k in [self.rbf, self.lin, self.k]:
            with k.tf_mode():
                k._K = tf.Session().run(k.K(X), feed_dict={x_free: k.get_free_state(), X: X_data})

        self.failUnless(np.allclose(self.rbf._K + self.lin._K, self.k._K))

    def test_asym(self):
        x_free = tf.placeholder('float64')
        [k.make_tf_array(x_free) for k in (self.rbf, self.lin, self.k)]
        X = tf.placeholder('float64')
        Z = tf.placeholder('float64')
        X_data = self.rng.randn(10, 1)
        Z_data = self.rng.randn(12, 1)
        for k in [self.rbf, self.lin, self.k]:
            with k.tf_mode():
                k._K = tf.Session().run(k.K(X), feed_dict={x_free: k.get_free_state(), X: X_data, Z: Z_data})

        self.failUnless(np.allclose(self.rbf._K + self.lin._K, self.k._K))


class TestWhite(unittest.TestCase):
    """
    The white kernel should not give the same result when called with k(X) and
    k(X, X)
    """
    def setUp(self):
        tf.reset_default_graph()
        self.k = GPflow.kernels.White(1)
        self.rng = np.random.RandomState(0)

    def test(self):
        x_free = tf.placeholder('float64')
        self.k.make_tf_array(x_free)
        X = tf.placeholder('float64')
        X_data = self.rng.randn(10, 1)
        with self.k.tf_mode():
            K_sym = tf.Session().run(self.k.K(X), feed_dict={x_free: self.k.get_free_state(), X: X_data})
            K_asym = tf.Session().run(self.k.K(X, X), feed_dict={x_free: self.k.get_free_state(), X: X_data})

        self.failIf(np.allclose(K_sym, K_asym))


class TestSlice(unittest.TestCase):
    """
    Make sure the results of a sliced kernel is the ame as an unsliced kernel
    with correctly sliced data...
    """
    def setUp(self):
        tf.reset_default_graph()
        self.rng = np.random.RandomState(0)
        self.k1 = GPflow.kernels.RBF(1, active_dims=[0])
        self.k2 = GPflow.kernels.RBF(1, active_dims=[1])
        self.k3 = GPflow.kernels.RBF(1)
        self.X = self.rng.randn(20, 2)
        self.Z = self.rng.randn(10, 2)

    def test_symm(self):
        K1 = self.k1.compute_K_symm(self.X)
        K2 = self.k2.compute_K_symm(self.X)
        K3 = self.k3.compute_K_symm(self.X[:, :1])
        K4 = self.k3.compute_K_symm(self.X[:, 1:])
        self.failUnless(np.allclose(K1, K3))
        self.failUnless(np.allclose(K2, K4))

    def test_asymm(self):
        K1 = self.k1.compute_K(self.X, self.Z)
        K2 = self.k2.compute_K(self.X, self.Z)
        K3 = self.k3.compute_K(self.X[:, :1], self.Z[:, :1])
        K4 = self.k3.compute_K(self.X[:, 1:], self.Z[:, 1:])
        self.failUnless(np.allclose(K1, K3))
        self.failUnless(np.allclose(K2, K4))


class TestProd(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.k1 = GPflow.kernels.Matern32(2)
        self.k2 = GPflow.kernels.Matern52(2, lengthscales=0.3)
        self.k3 = self.k1 * self.k2
        self.x_free = tf.placeholder(tf.float64)
        self.X = tf.placeholder(tf.float64, [30, 2])
        self.X_data = np.random.randn(30, 2)

    def test_prod(self):
        with self.k1.tf_mode():
            with self.k2.tf_mode():
                with self.k3.tf_mode():

                    self.k1.make_tf_array(self.x_free)
                    K1 = self.k1.K(self.X)
                    K1 = tf.Session().run(K1, feed_dict={self.X: self.X_data, self.x_free: self.k1.get_free_state()})

                    self.k2.make_tf_array(self.x_free)
                    K2 = self.k2.K(self.X)
                    K2 = tf.Session().run(K2, feed_dict={self.X: self.X_data, self.x_free: self.k2.get_free_state()})

                    self.k3.make_tf_array(self.x_free)
                    K3 = self.k3.K(self.X)
                    K3 = tf.Session().run(K3, feed_dict={self.X: self.X_data, self.x_free: self.k3.get_free_state()})
        self.failUnless(np.allclose(K1 * K2, K3))


class TestARDActiveProd(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.rng = np.random.RandomState(0)

        # k3 = k1 * k2
        self.k1 = GPflow.kernels.RBF(3, active_dims=[0, 1, 3], ARD=True)
        self.k2 = GPflow.kernels.RBF(1, active_dims=[2], ARD=True)
        self.k3 = GPflow.kernels.RBF(4, ARD=True)
        self.k1.lengthscales = np.array([3.4, 4.5, 5.6])
        self.k2.lengthscales = 6.7
        self.k3.lengthscales = np.array([3.4, 4.5, 6.7,  5.6])
        self.k3a = self.k1 * self.k2

        # make kernel functions in python
        self.x_free = tf.placeholder('float64')
        self.k3.make_tf_array(self.x_free)
        self.k3a.make_tf_array(self.x_free)
        self.X = tf.placeholder('float64', [50, 4])
        self.X_data = np.random.randn(50, 4)

    def test(self):
        with self.k3.tf_mode():
            with self.k3a.tf_mode():
                K1 = self.k3.K(self.X)
                K2 = self.k3a.K(self.X)
                K1 = tf.Session().run(K1, feed_dict={self.X: self.X_data, self.x_free: self.k3.get_free_state()})
                K2 = tf.Session().run(K2, feed_dict={self.X: self.X_data, self.x_free: self.k3a.get_free_state()})

        self.failUnless(np.allclose(K1, K2))


class TestKernNaming(unittest.TestCase):
    def test_no_nesting_1(self):
        k1 = GPflow.kernels.RBF(1)
        k2 = GPflow.kernels.Linear(2)
        k3 = k1 + k2
        k4 = GPflow.kernels.Matern32(1)
        k5 = k3 + k4
        self.failUnless(k5.rbf is k1)
        self.failUnless(k5.linear is k2)
        self.failUnless(k5.matern32 is k4)

    def test_no_nesting_2(self):
        k1 = GPflow.kernels.RBF(1) + GPflow.kernels.Linear(2)

        k2 = GPflow.kernels.Matern32(1) + GPflow.kernels.Matern52(2)

        k = k1 + k2
        self.failUnless(hasattr(k, 'rbf'))
        self.failUnless(hasattr(k, 'linear'))
        self.failUnless(hasattr(k, 'matern32'))
        self.failUnless(hasattr(k, 'matern52'))

    def test_simple(self):
        k1 = GPflow.kernels.RBF(1)
        k2 = GPflow.kernels.Linear(2)
        k = k1 + k2
        self.failUnless(k.rbf is k1)
        self.failUnless(k.linear is k2)

    def test_duplicates_1(self):
        k1 = GPflow.kernels.Matern32(1)
        k2 = GPflow.kernels.Matern32(43)
        k = k1 + k2
        self.failUnless(k.matern32_1 is k1)
        self.failUnless(k.matern32_2 is k2)

    def test_duplicates_2(self):
        k1 = GPflow.kernels.Matern32(1)
        k2 = GPflow.kernels.Matern32(2)
        k3 = GPflow.kernels.Matern32(3)
        k = k1 + k2 + k3
        self.failUnless(k.matern32_1 is k1)
        self.failUnless(k.matern32_2 is k2)
        self.failUnless(k.matern32_3 is k3)


class TestKernNamingProduct(unittest.TestCase):
    def test_no_nesting_1(self):
        k1 = GPflow.kernels.RBF(1)
        k2 = GPflow.kernels.Linear(2)
        k3 = k1 * k2
        k4 = GPflow.kernels.Matern32(1)
        k5 = k3 * k4
        self.failUnless(k5.rbf is k1)
        self.failUnless(k5.linear is k2)
        self.failUnless(k5.matern32 is k4)

    def test_no_nesting_2(self):
        k1 = GPflow.kernels.RBF(1) * GPflow.kernels.Linear(2)

        k2 = GPflow.kernels.Matern32(1) * GPflow.kernels.Matern52(2)

        k = k1 * k2
        self.failUnless(hasattr(k, 'rbf'))
        self.failUnless(hasattr(k, 'linear'))
        self.failUnless(hasattr(k, 'matern32'))
        self.failUnless(hasattr(k, 'matern52'))

    def test_simple(self):
        k1 = GPflow.kernels.RBF(1)
        k2 = GPflow.kernels.Linear(2)
        k = k1 * k2
        self.failUnless(k.rbf is k1)
        self.failUnless(k.linear is k2)

    def test_duplicates_1(self):
        k1 = GPflow.kernels.Matern32(1)
        k2 = GPflow.kernels.Matern32(43)
        k = k1 * k2
        self.failUnless(k.matern32_1 is k1)
        self.failUnless(k.matern32_2 is k2)

    def test_duplicates_2(self):
        k1 = GPflow.kernels.Matern32(1)
        k2 = GPflow.kernels.Matern32(2)
        k3 = GPflow.kernels.Matern32(3)
        k = k1 * k2 * k3
        self.failUnless(k.matern32_1 is k1)
        self.failUnless(k.matern32_2 is k2)
        self.failUnless(k.matern32_3 is k3)


class TestARDInit(unittest.TestCase):
    """
    For ARD kernels, make sure that kernels can be instantiated with a single
    lengthscale or a suitable array of lengthscales
    """
    def test_scalar(self):
        k1 = GPflow.kernels.RBF(3, lengthscales=2.3)
        k2 = GPflow.kernels.RBF(3, lengthscales=np.ones(3) * 2.3)
        self.assertTrue(np.all(k1.lengthscales.value == k2.lengthscales.value))


if __name__ == "__main__":
    unittest.main()
