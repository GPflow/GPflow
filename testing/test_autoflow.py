import gpflow
import tensorflow as tf
import numpy as np
import sys
import unittest


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


class TestNoArgs(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = NoArgsModel()
        self.m._compile()

    def test_return(self):
        self.assertTrue(np.allclose(self.m.function(), 3.))

    def test_kill(self):
        # make sure autoflow dicts are removed when _needs_recompile is set.
        keys = [k for k in self.m.__dict__.keys() if k.endswith('_AF_storage')]
        self.assertTrue(len(keys) == 0, msg="no AF storage should be present to start.")

        self.m.function()

        keys = [k for k in self.m.__dict__.keys() if k.endswith('_AF_storage')]
        self.assertTrue(len(keys) == 1, msg="AF storage should be present after function call.")

        self.m._needs_recompile = True

        keys = [k for k in self.m.__dict__.keys() if k.endswith('_AF_storage')]
        self.assertTrue(len(keys) == 0, msg="no AF storage should be present after recompile switch set.")


class AddModel(DumbModel):
    @gpflow.model.AutoFlow((tf.float64,), (tf.float64,))
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
        self.assertTrue(np.allclose(self.x + self.y, self.m.add(self.x, self.y)))


class IncrementModel(DumbModel):
    def __init__(self):
        DumbModel.__init__(self)
        self.a = gpflow.param.DataHolder(np.array([3.]))

    @gpflow.model.AutoFlow((tf.float64,))
    def inc(self, x):
        return x + self.a


class TestDataHolder(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = IncrementModel()
        rng = np.random.RandomState(0)
        self.x = rng.randn(10, 20)

    def test_add(self):
        self.assertTrue(np.allclose(self.x + 3, self.m.inc(self.x)))


class TestGPmodel(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)
        X, Y = rng.randn(2, 10, 1)
        self.m = gpflow.svgp.SVGP(X, Y, kern=gpflow.kernels.Matern32(1),
                                  likelihood=gpflow.likelihoods.StudentT(),
                                  Z=X[::2].copy())
        self.Xtest = np.random.randn(100, 1)
        self.Ytest = np.random.randn(100, 1)

    def test_predict_f(self):
        mu, var = self.m.predict_f(self.Xtest)

    def test_predict_y(self):
        mu, var = self.m.predict_y(self.Xtest)

    def test_predict_density(self):
        self.m.predict_density(self.Xtest, self.Ytest)

    def test_multiple_AFs(self):
        self.m.compute_log_likelihood()
        self.m.compute_log_prior()
        self.m.compute_log_likelihood()


class TestResetGraph(unittest.TestCase):
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


class TestFixAndPredict(unittest.TestCase):
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
        self.m._compile()
        self.m.kern.variance.fixed = True
        _, _ = self.m.predict_f(self.m.X.value)


class TestSVGP(unittest.TestCase):
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


class CombinedAutoflowModel(DumbModel):
    @gpflow.model.AutoFlow(b=(tf.float64,))
    def multiply(self, a, b):
        return a * b

    @gpflow.model.AutoFlow((tf.float64,), x=(tf.float64,))
    def add(self, a, b, x=-1., only_a_and_b=False, **zs):
        if only_a_and_b:
            return a + b
        else:
            result = a + b + x
            for z in zs.values():
                result += z
            return result

    @gpflow.model.AutoFlow(x=(tf.float64,), b=(tf.float64,))
    def subtract(self, x, **zs):
        result = x
        for z in zs.values():
            result -= z
        return result

    if sys.version_info >= (3,):
        @gpflow.model.AutoFlow(a=(tf.float64,))
        def divide(self, a, *cs):
            result = a
            for c in cs:
                result /= c
            return result


class TestMixedArgs(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = CombinedAutoflowModel()
        self.m._compile()
        rng = np.random.RandomState(0)
        self.a = rng.randn(10, 20)
        self.b = rng.randn(10, 20)
        self.x = rng.randn(10, 20)

    def test_simple(self):
        self.assertTrue(np.allclose(1. * self.b,
                                    self.m.multiply(1., self.b)))
        self.assertTrue(np.allclose(1 * self.b,
                                    self.m.multiply(1, self.b)))
        self.assertTrue(np.allclose(2. * self.b,
                                    self.m.multiply(2., self.b)))
        self.assertTrue(np.allclose(self.a[0] * self.b,
                                    self.m.multiply(tuple(self.a[0]), self.b)))

    def test_varkwargs(self):
        self.assertTrue(np.allclose(3 - 1. - 2.,
                                    self.m.subtract(x=3., a=1., b=2.)))
        self.assertTrue(np.allclose(self.x - 17. - self.b,
                                    self.m.subtract(a=17., b=self.b, x=self.x)))

    def test_combined(self):
        self.assertTrue(np.allclose(self.a + 1.,
                                    self.m.add(self.a, 1., only_a_and_b=True)))
        self.assertTrue(np.allclose(self.a + 1.,
                                    self.m.add(a=self.a, b=1., only_a_and_b=True)))
        self.assertTrue(np.allclose(self.a + 1. + self.x + 4.,
                                    self.m.add(self.a, 1., x=self.x, y=4.)))

    if sys.version_info >= (3,):
        def test_varargs(self):
            self.assertTrue(np.allclose(1. / 2. / 3. / -4.,
                                        self.m.divide(1., 2., 3., -4.)))
            self.assertTrue(np.allclose(self.a / 2. / 3. / -4.,
                                        self.m.divide(self.a, 2., 3., -4.)))


class CachedModel(DumbModel):
    @gpflow.model.AutoFlow(a=(tf.float64,))
    def add(self, a, b, double_result=False):
        if double_result:
            return 2 * (a + b)
        else:
            return a + b


class TestGraphCaching(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = CachedModel()
        self.m._compile()
        rng = np.random.RandomState(0)
        self.a = rng.randn(10, 20)
        self.b = rng.randn(10, 20)

    def test_caching(self):
        def number_of_cached_graphs(model):
            keys = [k for k in self.m.__dict__.keys() if k.endswith('_AF_storage')]
            return len(keys)

        self.assertTrue(number_of_cached_graphs(self.m) == 0)

        self.m.add(self.a, 1.)
        self.assertTrue(number_of_cached_graphs(self.m) == 1)

        self.m.add(self.b, 1.)
        self.assertTrue(number_of_cached_graphs(self.m) == 1)

        self.m.add(self.a, 2.)
        self.assertTrue(number_of_cached_graphs(self.m) == 2)

        self.m.add(self.a, 2., double_result=True)
        self.assertTrue(number_of_cached_graphs(self.m) == 3)

        self.m.add(self.b, 2., double_result=True)
        self.assertTrue(number_of_cached_graphs(self.m) == 3)


if __name__ == "__main__":
    unittest.main()
