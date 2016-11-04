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


class TestPsi1D_GH(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState()
        self.kerns = [GPflow.kernels.PeriodicKernel(1),
                      GPflow.kernels.RBF(1), GPflow.kernels.Linear(1),
                      GPflow.kernels.RBF(1)+GPflow.kernels.Linear(1),
                      GPflow.kernels.Linear(1)+GPflow.kernels.RBF(1)]
        self.z = rng.randn(1, 1)
        self.mu = rng.randn()
        self.var = rng.rand()

    def test(self):
        for k in self.kerns:
            f = lambda x: k.compute_K(self.z, np.atleast_2d(x)).squeeze()
            result_numeric = do_GH_quadratre(f, self.mu, self.var)
            result_tf = PsiComputer(k).psi1(np.atleast_2d(self.mu), np.atleast_2d(self.var), self.z)
            print(k.__class__)
            print(result_tf, result_numeric)

            self.assertTrue(np.allclose(result_numeric, result_tf))

            # TODO: should check psi0 and psi2 stats as well


class TestPsi2D_GH(unittest.TestCase):
    ''' Test psi statistics calculations with kernels on different active dimensions '''
    def setUp(self):
        rng = np.random.RandomState()
        self.kerns = [GPflow.kernels.RBF(1, active_dims=[0])+GPflow.kernels.RBF(1, active_dims=[1]),
# FAILS                      GPflow.kernels.RBF(1, active_dims=[0])+GPflow.kernels.PeriodicKernel(1, active_dims=[1]),
                      GPflow.kernels.RBF(1, active_dims=[0])*GPflow.kernels.Linear(1, active_dims=[1])]
        self.z = rng.randn(1, 2)
        self.mu = rng.randn(1, 2)
        self.var = rng.rand(1, 2)

    def test(self):
        for kcomposite in self.kerns:
            print('init', kcomposite.__class__)
            self.assertTrue(GPflow.kernel_expectations.on_separate_dimensions(kcomposite.kern_list))  
            result_numeric = int(isinstance(kcomposite, GPflow.kernels.Prod))  # init with 0 for add, 1 for prod
            for ik,k in enumerate(kcomposite.kern_list):
                f = lambda x: k.compute_K(self.z, np.ones((1, 2)) * np.atleast_2d(x)).squeeze()  # other dim ignored by kernel
                r = do_GH_quadratre(f, self.mu[:, ik], self.var[:, ik])
                if(isinstance(kcomposite, GPflow.kernels.Prod)):
                    result_numeric *= r
                else:
                    result_numeric += r

            result_tf = PsiComputer(kcomposite).psi1(np.atleast_2d(self.mu), np.atleast_2d(self.var), self.z)
            print(kcomposite.__class__)
            print(result_tf, result_numeric)

            self.assertTrue(np.allclose(result_numeric, result_tf))


class TestActiveDimensionChecks(unittest.TestCase):
    def test(self):
        k = GPflow.kernels.PeriodicKernel(1)
        self.assertTrue(GPflow.kernel_expectations.is_one_dimensional(k))

        k = GPflow.kernels.RBF(1, active_dims=[0])
        self.assertTrue(GPflow.kernel_expectations.is_one_dimensional(k))

        k = GPflow.kernels.RBF(2, active_dims=[0,2])
        self.assertFalse(GPflow.kernel_expectations.is_one_dimensional(k))

        k = GPflow.kernels.RBF(2, active_dims=[0,2]) + GPflow.kernels.PeriodicKernel(2, active_dims=[0,2])
        self.assertFalse(GPflow.kernel_expectations.on_separate_dimensions(k.kern_list))

        k = GPflow.kernels.RBF(2, active_dims=[2,3]) + GPflow.kernels.PeriodicKernel(2, active_dims=[0,1])
        self.assertTrue(GPflow.kernel_expectations.on_separate_dimensions(k.kern_list))

        k = GPflow.kernels.RBF(4) + GPflow.kernels.PeriodicKernel(4)
        self.assertFalse(GPflow.kernel_expectations.on_separate_dimensions(k.kern_list))


if __name__ == "__main__":
    unittest.main()
