import GPflow
import tensorflow as tf
import numpy as np
import unittest


class DumbModel(GPflow.model.Model):
    def __init__(self):
        GPflow.model.Model.__init__(self)
        self.a = GPflow.param.Param(3.)

    def build_likelihood(self):
        return -tf.square(self.a)


class NoArgsModel(DumbModel):
    @GPflow.model.AutoFlow()
    def function(self):
        return self.a


class TestNoArgs(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = NoArgsModel()
        self.m._compile()

    def test_return(self):
        self.failUnless(np.allclose(self.m.function(), 3.))


class AddModel(DumbModel):
    @GPflow.model.AutoFlow((tf.float64,), (tf.float64,))
    def add(self, x, y):
        return tf.add(x, y)


class TestShareArgs(unittest.TestCase):
    """
    This is designed to replicate bug #85, where having two models caused
    autoflow functions to fail because the tf_args were shared over the
    instances.
    """
    def setUp(self):
        tf.reset_default_graph()
        self.m1 = AddModel()
        self.m1._compile()
        self.m2 = AddModel()
        self.m2._compile()
        rng = np.random.RandomState(0)
        self.x = rng.randn(10, 20)
        self.y = rng.randn(10, 20)

    def test_share_args(self):
        self.m1.add(self.x, self.y)
        self.m2.add(self.x, self.y)
        self.m1.add(self.x, self.y)


class TestAdd(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = AddModel()
        self.m._compile()
        rng = np.random.RandomState(0)
        self.x = rng.randn(10, 20)
        self.y = rng.randn(10, 20)

    def test_add(self):
        self.failUnless(np.allclose(self.x + self.y, self.m.add(self.x, self.y)))


class TestGPmodel(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        self.m = GPflow.svgp.SVGP(X, Y, kern=GPflow.kernels.Matern32(1),
                                  likelihood=GPflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        self.Xtest = np.random.randn(100, 1)
        self.Ytest = np.random.randn(100, 1)

    def test_predict_f(self):
        mu, var = self.m.predict_f(self.Xtest)

    def test_predict_y(self):
        mu, var = self.m.predict_y(self.Xtest)

    def test_predict_density(self):
        self.m.predict_density(self.Xtest, self.Ytest)


class TestResetGraph(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        k = GPflow.kernels.Matern32(1)
        X, Y = np.random.randn(2, 10, 1)
        self.Xnew = np.random.randn(5, 1)
        self.m = GPflow.gpr.GPR(X, Y, kern=k)

    def test(self):
        mu, var = self.m.predict_f(self.Xnew)
        tf.reset_default_graph()
        mu1, var1 = self.m.predict_f(self.Xnew)


class TestFixAndPredict(unittest.TestCase):
    """
    Bug #54 says that if a model parameter is fixed  between calls to predict
    (an autoflow fn) then the second call fails. This test ensures replicates
    that and ensures that the bugfix remains in furure.
    """

    def setUp(self):
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        self.m = GPflow.svgp.SVGP(X, Y, kern=GPflow.kernels.Matern32(1),
                                  likelihood=GPflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        self.Xtest = np.random.randn(100, 1)
        self.Ytest = np.random.randn(100, 1)

    def test(self):
        self.m._compile()
        self.m.kern.variance.fixed = True
        _, _ = self.m.predict_f(self.m.X)


class TestSVGP(unittest.TestCase):
    """
    This replicates Alex's code from bug #99
    """
    def test(self):
        rng = np.random.RandomState(1)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        Z = rng.randn(3, 1)
        model = GPflow.svgp.SVGP(X=X, Y=Y, kern=GPflow.kernels.RBF(1), likelihood=GPflow.likelihoods.Gaussian(), Z=Z)
        model.compute_log_likelihood()


if __name__ == "__main__":
    unittest.main()
