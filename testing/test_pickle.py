import unittest
import GPflow
import numpy as np
import pickle


class TestPickleEmpty(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()

    def test(self):
        s = pickle.dumps(self.m)
        pickle.loads(s)


class TestPickleSimple(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()
        self.m.p1 = GPflow.param.Param(np.random.randn(3, 2))
        self.m.p2 = GPflow.param.Param(np.random.randn(10))

    def test(self):
        s = pickle.dumps(self.m)
        m2 = pickle.loads(s)
        self.assertTrue(m2.p1._parent is m2)
        self.assertTrue(m2.p2._parent is m2)


class TestPickleGPR(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        self.m = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.RBF(1))

    def test(self):
        s1 = pickle.dumps(self.m)  # the model without running _compile
        self.m._compile()
        s2 = pickle.dumps(self.m)  # the model after _compile

        # reload the model
        m1 = pickle.loads(s1)
        m2 = pickle.loads(s2)

        # make sure the log likelihoods still match
        l1 = self.m.compute_log_likelihood()
        l2 = m1.compute_log_likelihood()
        l3 = m2.compute_log_likelihood()
        self.assertTrue(l1 == l2 == l3)


class TestPickleSVGP(unittest.TestCase):
    """
    Like the TestPickleGPR test, but with svgp (since it has extra tf variables
    for minibatching)
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        Z = rng.randn(5, 1)
        self.m = GPflow.svgp.SVGP(X, Y, Z=Z,
                                  likelihood=GPflow.likelihoods.Gaussian(),
                                  kern=GPflow.kernels.RBF(1))

    def test(self):
        s1 = pickle.dumps(self.m)  # the model without running _compile
        self.m._compile()
        s2 = pickle.dumps(self.m)  # the model after _compile

        # reload the model
        m1 = pickle.loads(s1)
        m2 = pickle.loads(s2)

        # make sure the log likelihoods still match
        l1 = self.m.compute_log_likelihood()
        l2 = m1.compute_log_likelihood()
        l3 = m2.compute_log_likelihood()
        self.assertTrue(l1 == l2 == l3)


class TestPickleAndDict(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        self.m = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.RBF(1))

    def test(self):
        # pickle and reload the model
        s1 = pickle.dumps(self.m)
        m1 = pickle.loads(s1)

        d1 = self.m.to_dict()
        d2 = m1.to_dict()
        for key, val in d1.iteritems():
            assert np.all(val==d2[key])


if __name__ == "__main__":
    unittest.main()
