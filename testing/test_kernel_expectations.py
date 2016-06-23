import GPflow
import numpy as np
import tensorflow as tf
import unittest
hermgauss = np.polynomial.hermite.hermgauss


def do_GH_quadratre(f, mu, var, numpoints=20):
    gh_x, gh_w = hermgauss(numpoints)
    gh_w /= np.sqrt(np.pi)
    X = gh_x * np.sqrt(2.0 * var) + mu
    return sum([f(xi)*wi for xi, wi in zip(X, gh_w)])


class PsiComputer(GPflow.param.Parameterized):
    def __init__(self, kern):
        GPflow.param.Parameterized.__init__(self)
        self.kern = kern

    @GPflow.param.AutoFlow((tf.float64,), (tf.float64,), (tf.float64,))
    def psi1(self, mu, S, Z):
        return GPflow.kernel_expectations.build_psi_stats(Z, self.kern, mu, S)[1]

    @GPflow.param.AutoFlow((tf.float64,), (tf.float64,), (tf.float64,))
    def psi2(self, mu, S, Z):
        return GPflow.kernel_expectations.build_psi_stats(Z, self.kern, mu, S)[2]


class TestPsi1_GH(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState()
        self.kerns = [GPflow.kernels.RBF(1), GPflow.kernels.Linear(1), GPflow.kernels.RBF(1)+GPflow.kernels.Linear(1)]
        # self.kerns = [GPflow.kernels.Linear(1)]#, GPflow.kernels.Linear(1)]
        self.z = rng.randn(1, 1)
        self.mu = rng.randn()
        self.var = rng.rand()

    def test(self):
        for k in self.kerns:
            f = lambda x: k.compute_K(self.z, np.atleast_2d(x)).squeeze()
            result_numeric = do_GH_quadratre(f, self.mu, self.var)
            result_tf = PsiComputer(k).psi1(np.atleast_2d(self.mu), np.atleast_2d(self.var), self.z)
            print k.__class__
            print result_tf, result_numeric

            self.assertTrue(np.allclose(result_numeric, result_tf))

if __name__ == "__main__":
    unittest.main()
