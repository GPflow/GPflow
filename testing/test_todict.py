import unittest
import GPflow
import numpy as np
import pickle


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

        d1 = self.m.get_parameter_dict()
        d2 = m1.get_parameter_dict()
        for key, val in d1.items():
            assert np.all(val == d2[key])


if __name__ == "__main__":
    unittest.main()
