import gpflow
import tensorflow as tf
import numpy as np
import unittest

from testing.gpflow_testcase import GPflowTestCase


class DumbModel(gpflow.model.Model):
    def __init__(self):
        gpflow.model.Model.__init__(self)
        self.a = gpflow.param.Param(3.)

    def build_likelihood(self):
        return -tf.square(self.a)


class NoArgsModel(DumbModel):
    @gpflow.model.AutoFlow()
    def function(self):
        return self.a


class TestNoArgs(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.m = NoArgsModel()
            self.m.compile()

    def test_return(self):
        with self.test_session():
            self.assertTrue(np.allclose(self.m.function(), 3.))

    def test_kill(self):
        # make sure autoflow dicts are removed when _needs_recompile is set.
        with self.test_session():
            keys = [k for k in self.m.__dict__.keys() if k[-11:] == '_AF_storage']
            self.assertTrue(len(keys) == 0, msg="no AF storage should be present to start.")

            self.m.function()

            keys = [k for k in self.m.__dict__.keys() if k[-11:] == '_AF_storage']
            self.assertTrue(len(keys) == 1, msg="AF storage should be present after function call.")

            self.m._needs_recompile = True

            keys = [k for k in self.m.__dict__.keys() if k[-11:] == '_AF_storage']
            self.assertTrue(len(keys) == 0, msg="no AF storage should be present after recompile switch set.")


class AddModel(DumbModel):
    @gpflow.model.AutoFlow((tf.float64,), (tf.float64,))
    def add(self, x, y):
        return tf.add(x, y)


class TestShareArgs(GPflowTestCase):
    """
    This is designed to replicate bug #85, where having two models caused
    autoflow functions to fail because the tf_args were shared over the
    instances.
    """
    def setUp(self):
        with self.test_session():
            self.m1 = AddModel()
            self.m1.compile()
            self.m2 = AddModel()
            self.m2.compile()
            rng = np.random.RandomState(0)
            self.x = rng.randn(10, 20)
            self.y = rng.randn(10, 20)

    def test_share_args(self):
        with self.test_session():
            self.m1.add(self.x, self.y)
            self.m2.add(self.x, self.y)
            self.m1.add(self.x, self.y)


class TestAutoFlowSessionGraphArguments(GPflowTestCase):
    """AutoFlow tests for external session and graph."""

    def setUp(self):
        self.models = [AddModel() for _ in range(5)]
        self.session = tf.Session()
        self.graph = tf.Graph()
        self.x = np.array([1., 1., 1.])
        self.y = np.array([1., 2., 3.])

    def test_wrong_arguments(self):
        """Wrong arguments for AutoFlow wrapped function."""
        m1 = self.models[0]
        self.assertRaises(ValueError, m1.add, [1.], [1.],
                          unknown1='argument1')
        self.assertRaises(ValueError, m1.add, [1.], [1.],
                          graph=tf.Graph(), unknown1='argument1')
        self.assertRaises(ValueError, m1.add, [1.], [1.],
                          session=self.session, unknown2='argument2')
        self.assertRaises(ValueError, m1.add, [1.], [1.],
                          graph=tf.Graph(), session=tf.Session())

    def test_storage_properties(self):
        """External graph and session passed to AutoFlow."""

        m1, m2, m3, m4, m5 = self.models
        storage_name = '_add_AF_storage'
        m1.add(self.x, self.y)
        m2.add(self.x, self.y, session=self.session)
        m3.add(self.x, self.y, graph=self.graph)

        with self.graph.as_default():
            m4.add(self.x, self.y)

        with self.test_session() as sess_default:
            m5.add(self.x, self.y)

        sessions = [getattr(m, storage_name)['session'] for m in self.models]
        sess1, sess2, sess3, sess4, sess5 = sessions
        sessions_set = set(map(str, sessions))
        self.assertEqual(len(sessions_set), 5)
        self.assertEqual(sess1.graph, sess2.graph)
        self.assertEqual(sess3.graph, sess4.graph)
        self.assertEqual(sess5.graph, sess_default.graph)
        self.assertEqual(sess5, sess_default)

        m2.add(self.x, self.y)
        sess2_second_run = getattr(m2, storage_name)['session']
        self.assertTrue(isinstance(sess2_second_run, tf.Session))
        self.assertEqual(sess2, sess2_second_run)

        m4.add(self.x, self.y, graph=tf.get_default_graph())
        sess4_second_run = getattr(m4, storage_name)['session']
        self.assertTrue(isinstance(sess4_second_run, tf.Session))
        self.assertNotEqual(sess4, sess4_second_run)

        with self.test_session():
            m5.add(self.x, self.y, graph=sess_default.graph)
        sess5_second_run = getattr(m5, storage_name)['session']
        self.assertTrue(isinstance(sess5_second_run, tf.Session))
        self.assertEqual(sess5, sess5_second_run)
        self.assertEqual(sess5_second_run, sess_default)

        m5.add(self.x, self.y, graph=sess_default.graph)
        sess5_third_run = getattr(m5, storage_name)['session']
        self.assertTrue(isinstance(sess5_third_run, tf.Session))
        self.assertNotEqual(sess5, sess5_third_run)
        self.assertNotEqual(sess5_third_run, sess_default)

        sess5_third_run.close()

        _ = [sess.close() for sess in sessions]
        _ = [getattr(m, storage_name)['session'].close() for m in self.models]


    def test_autoflow_results(self):
        """AutoFlow computation results for external session and graph."""
        expected = self.x + self.y

        m1, m2, m3, m4, m5 = self.models

        def assert_add(model, **kwargs):
            result = model.add(self.x, self.y, **kwargs)
            self.assertTrue(np.all(result == expected))

        assert_add(m1)
        assert_add(m2, session=self.session)
        assert_add(m3, graph=self.graph)

        with self.graph.as_default():
            assert_add(m4)

        with self.test_session():
            assert_add(m5)


class TestAdd(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.m = AddModel()
            self.m.compile()
            rng = np.random.RandomState(0)
            self.x = rng.randn(10, 20)
            self.y = rng.randn(10, 20)

    def test_add(self):
        with self.test_session():
            self.assertTrue(np.allclose(self.x + self.y, self.m.add(self.x, self.y)))


class IncrementModel(DumbModel):
    def __init__(self):
        DumbModel.__init__(self)
        self.a = gpflow.param.DataHolder(np.array([3.]))

    @gpflow.model.AutoFlow((tf.float64,))
    def inc(self, x):
        return x + self.a


class TestDataHolder(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.m = IncrementModel()
            rng = np.random.RandomState(0)
            self.x = rng.randn(10, 20)

    def test_add(self):
        with self.test_session():
            self.assertTrue(np.allclose(self.x + 3, self.m.inc(self.x)))


class TestGPmodel(GPflowTestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        self.m = gpflow.svgp.SVGP(X, Y, kern=gpflow.kernels.Matern32(1),
                                  likelihood=gpflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        self.Xtest = np.random.randn(100, 1)
        self.Ytest = np.random.randn(100, 1)

    def test_predict_f(self):
        with self.test_session():
            mu, var = self.m.predict_f(self.Xtest)

    def test_predict_y(self):
        with self.test_session():
            mu, var = self.m.predict_y(self.Xtest)

    def test_predict_density(self):
        with self.test_session():
            self.m.predict_density(self.Xtest, self.Ytest)

    def test_multiple_AFs(self):
        with self.test_session():
            self.m.compute_log_likelihood()
            self.m.compute_log_prior()
            self.m.compute_log_likelihood()


class TestResetGraph(GPflowTestCase):
    def setUp(self):
        tf.reset_default_graph()
        k = gpflow.kernels.Matern32(1)
        X, Y = np.random.randn(2, 10, 1)
        self.Xnew = np.random.randn(5, 1)
        self.m = gpflow.gpr.GPR(X, Y, kern=k)

    def test(self):
        mu, var = self.m.predict_f(self.Xnew)
        tf.reset_default_graph()
        mu1, var1 = self.m.predict_f(self.Xnew)


class TestFixAndPredict(GPflowTestCase):
    """
    Bug #54 says that if a model parameter is fixed  between calls to predict
    (an autoflow fn) then the second call fails. This test ensures replicates
    that and ensures that the bugfix remains in furure.
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        self.m = gpflow.svgp.SVGP(X, Y, kern=gpflow.kernels.Matern32(1),
                                  likelihood=gpflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        self.Xtest = np.random.randn(100, 1)
        self.Ytest = np.random.randn(100, 1)

    def test(self):
        self.m.compile()
        self.m.kern.variance.fixed = True
        _, _ = self.m.predict_f(self.m.X.value)


class TestSVGP(GPflowTestCase):
    """
    This replicates Alex's code from bug #99
    """
    def test(self):
        rng = np.random.RandomState(1)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        Z = rng.randn(3, 1)
        model = gpflow.svgp.SVGP(X=X, Y=Y, kern=gpflow.kernels.RBF(1), likelihood=gpflow.likelihoods.Gaussian(), Z=Z)
        model.compute_log_likelihood()


if __name__ == "__main__":
    unittest.main()
