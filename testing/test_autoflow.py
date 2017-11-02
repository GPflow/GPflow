import unittest
import tensorflow as tf

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

import gpflow
from gpflow.test_util import GPflowTestCase
from gpflow.core import AutoFlow


class DumbModel(gpflow.models.Model):
    def __init__(self):
        gpflow.models.Model.__init__(self)
        self.a = gpflow.Param(3.)

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return -tf.square(self.a)


class NoArgsModel(DumbModel):
    @gpflow.autoflow()
    @gpflow.params_as_tensors
    def function(self):
        return self.a

    @gpflow.autoflow()
    @gpflow.params_as_tensors
    def function_another(self):
        return self.a + 1.0

class TestNoArgs(GPflowTestCase):
    def test_autoflow_functioning(self):
        with self.test_context():
            m = NoArgsModel()
            m.compile()

            def get_keys():
                return [k for k in m.__dict__ if k.startswith(AutoFlow.__autoflow_prefix__)]

            assert_allclose(m.function(), 3.)
            assert_allclose(m.function_another(), 4.)
            keys = get_keys()
            self.assertEqual(len(keys), 2)

            first_key = keys[0]
            AutoFlow.clear_autoflow(m, name=first_key)
            self.assertEqual(len(get_keys()), 1)
            assert_allclose(m.function(), 3.)
            self.assertEqual(len(get_keys()), 2)

            second_key = keys[1]
            AutoFlow.clear_autoflow(m, name=second_key)
            self.assertEqual(len(get_keys()), 1)
            assert_allclose(m.function_another(), 4.)
            self.assertEqual(len(get_keys()), 2)

            AutoFlow.clear_autoflow(m, name=first_key)
            AutoFlow.clear_autoflow(m, name=second_key)
            self.assertEqual(len(get_keys()), 0)
            assert_allclose(m.function(), 3.)
            assert_allclose(m.function_another(), 4.)
            self.assertEqual(len(get_keys()), 2)


            AutoFlow.clear_autoflow(m)
            self.assertEqual(len(get_keys()), 0)


class AddModel(DumbModel):
    @gpflow.autoflow((tf.float64,), (tf.float64,))
    @gpflow.params_as_tensors
    def add(self, x, y):
        return tf.add(x, y)


class TestShareArgs(GPflowTestCase):
    """
    This is designed to replicate bug #85, where having two models caused
    autoflow functions to fail because the tf_args were shared over the
    instances.
    """
    def setUp(self):
        self.m1 = AddModel()
        self.m2 = AddModel()

    def test_share_args(self):
        with self.test_context():
            self.m1.compile()
            self.m2.compile()
            rng = np.random.RandomState(0)
            x = rng.randn(10, 20)
            y = rng.randn(10, 20)
            ans = x + y
            self.m1.add(x, y)
            self.m2.add(x, y)
            assert_almost_equal(self.m1.add(x, y), ans)
            assert_almost_equal(self.m2.add(x, y), ans)
            assert_almost_equal(self.m1.add(y, x), ans)


class IncrementModel(DumbModel):
    def __init__(self):
        DumbModel.__init__(self)
        self.b = gpflow.DataHolder(np.array([3.]))

    @gpflow.autoflow((tf.float64,))
    @gpflow.params_as_tensors
    def inc(self, x):
        return x + self.b


class TestDataHolder(GPflowTestCase):
    def test_add(self):
        with self.test_context():
            m = IncrementModel()
            x = np.random.randn(10, 20)
            m.compile()
            assert_almost_equal(x + m.a.read_value(), m.inc(x))


class TestGPmodel(GPflowTestCase):
    def setup(self):
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        m = gpflow.models.SVGP(X, Y,
                               kern=gpflow.kernels.Matern32(1),
                               likelihood=gpflow.likelihoods.StudentT(),
                               Z=X[::2].copy())
        m.compile()
        xnew = np.random.randn(100, 1)
        ynew = np.random.randn(100, 1)
        return m, xnew, ynew

    def test_predict_f(self):
        with self.test_context():
            m, x, _y = self.setup()
            _mu, _var = m.predict_f(x)

    def test_predict_y(self):
        with self.test_context():
            m, x, _y = self.setup()
            _mu, _var = m.predict_y(x)

    def test_predict_density(self):
        with self.test_context():
            m, x, y = self.setup()
            m.predict_density(x, y)

    def test_multiple_AFs(self):
        with self.test_context():
            m, _x, _y = self.setup()
            m.compute_log_likelihood()
            m.compute_log_prior()
            m.compute_log_likelihood()


class TestResetGraph(GPflowTestCase):
    def setup(self):
        k = gpflow.kernels.Matern32(1)
        X, Y = np.random.randn(2, 10, 1)
        xnew = np.random.randn(5, 1)
        m = gpflow.models.GPR(X, Y, kern=k)
        session = tf.Session(graph=tf.Graph())
        m.compile(session=session)
        return m, xnew

    def test_reset_graph(self):
        m, x = self.setup()
        mu0, var0 = m.predict_f(x)
        tf.reset_default_graph()
        mu1, var1 = m.predict_f(x)
        assert_almost_equal(mu0, mu1)
        assert_almost_equal(var0, var1)


class TestFixAndPredict(GPflowTestCase):
    """
    Bug #54 says that if a model parameter is fixed  between calls to predict
    (an autoflow fn) then the second call fails. This test ensures replicates
    that and ensures that the bugfix remains in furure.
    """

    def setup(self):
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        m = gpflow.models.SVGP(X, Y, kern=gpflow.kernels.Matern32(1),
                                  likelihood=gpflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        xtest = np.random.randn(100, 1)
        ytest = np.random.randn(100, 1)
        return m, xtest, ytest

    def test(self):
        with self.test_context():
            m, x, y = self.setup()
            m.compile()
            m.kern.variance.trainable = False
            _, _ = m.predict_f(m.X.read_value())


class TestSVGP(GPflowTestCase):
    """
    This replicates Alex's code from bug #99
    """
    def test(self):
        rng = np.random.RandomState(1)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        Z = rng.randn(3, 1)
        model = gpflow.models.SVGP(
            X=X, Y=Y, kern=gpflow.kernels.RBF(1),
            likelihood=gpflow.likelihoods.Gaussian(), Z=Z)
        model.compile()
        model.compute_log_likelihood()


if __name__ == "__main__":
    unittest.main()
