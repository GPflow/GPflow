import unittest
import numpy as np
import numpy.random as rnd
from GPflow import kernels
from GPflow import ekernels
import GPflow.etransforms

rnd.seed(0)


class TestKernExpDelta(unittest.TestCase):
    """
    Check whether the normal kernel matrix is recovered if a delta distribution is used. First initial test which should
    indicate whether things work or not.
    """

    def setUp(self):
        self.D = 2
        self.Xmu = rnd.rand(10, self.D)
        self.Z = rnd.rand(4, self.D)
        self.Xcov = np.zeros((self.Xmu.shape[0], self.D, self.D))
        self.Xcovc = np.zeros((self.Xmu.shape[0], self.D, self.D))
        k1 = ekernels.RBF(self.D, ARD=True)
        k1.lengthscales = rnd.rand(2) + [0.5, 1.5]
        k1.variance = 0.3 + rnd.rand()
        klin = ekernels.Linear(self.D, variance=0.3 + rnd.rand())
        self.kernels = [k1, klin]

    def test_eKzxKxz(self):
        for k in self.kernels:
            psi2 = k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov)
            kernmat = k.compute_K(self.Z, self.Xmu)  # MxN
            kernouter = np.einsum('in,jn->nij', kernmat, kernmat)
            self.assertTrue(np.allclose(kernouter, psi2))

    def test_eKdiag(self):
        for k in self.kernels:
            kdiag = k.compute_eKdiag(self.Xmu, self.Xcov)
            orig = k.compute_Kdiag(self.Xmu)
            self.assertTrue(np.allclose(orig, kdiag))

    def test_exKxz(self):
        covall = np.array([self.Xcov, self.Xcovc])
        for k in self.kernels:
            if type(k) is ekernels.Linear:
                continue
            exKxz = k.compute_exKxz(self.Z, self.Xmu, covall)
            Kxz = k.compute_K(self.Xmu[:-1, :], self.Z)  # NxM
            xKxz = np.einsum('nm,nd->nmd', Kxz, self.Xmu[1:, :])
            self.assertTrue(np.allclose(xKxz, exKxz))

    def test_Kxz(self):
        for k in self.kernels:
            psi1 = k.compute_eKxz(self.Z, self.Xmu, self.Xcov)
            kernmat = k.compute_K(self.Z, self.Xmu)  # MxN
            self.assertTrue(np.allclose(kernmat, psi1.T))


class TestKernExpQuadrature(unittest.TestCase):
    _threshold = 0.5

    def setUp(self):
        self.N = 4
        self.D = 2
        self.Xmu = rnd.rand(self.N, self.D)
        self.Z = rnd.rand(2, self.D)

        unconstrained = rnd.randn(self.N, 2 * self.D, self.D)
        t = GPflow.etransforms.TriDiagonalBlockRep()
        self.Xcov = t.forward(unconstrained)

        ekernel_classes = [ekernels.RBF, ekernels.Linear]
        kernel_classes = [kernels.RBF, kernels.Linear]
        params = [(self.D, 0.3 + rnd.rand(), rnd.rand(2) + [0.5, 1.5], None, True),
                  (self.D, 0.3 + rnd.rand(), None)]
        self.ekernels = [c(*p) for c, p in zip(ekernel_classes, params)]
        self.kernels = [c(*p) for c, p in zip(kernel_classes, params)]

    def _assert_pdeq(self, a, b, k=None):
        pdmax = np.max((a / b - 1) * 100)
        self.assertTrue(pdmax < self._threshold, msg="Percentage difference above threshold: %f\n"
                                                     "On kernel: %s" % (pdmax, str(type(k))))

    def test_eKdiag(self):
        for k, ek in zip(self.kernels, self.ekernels):
            a = k.compute_eKdiag(self.Xmu, self.Xcov[0, :, :, :])
            b = ek.compute_eKdiag(self.Xmu, self.Xcov[0, :, :, :])
            self._assert_pdeq(a, b, k)

    def test_eKxz(self):
        for k, ek in zip(self.kernels, self.ekernels):
            a = k.compute_eKxz(self.Z, self.Xmu, self.Xcov[0, :, :, :])
            b = ek.compute_eKxz(self.Z, self.Xmu, self.Xcov[0, :, :, :])
            self._assert_pdeq(a, b, k)

    def test_eKzxKxz(self):
        for k, ek in zip(self.kernels, self.ekernels):
            k._kill_autoflow()
            k.num_gauss_hermite_points = 150
            a = k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov[0, :, :, :])
            b = ek.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov[0, :, :, :])
            self._assert_pdeq(a, b, k)

    def test_exKxz(self):
        for k, ek in zip(self.kernels, self.ekernels):
            k._kill_autoflow()
            k.num_gauss_hermite_points = 30
            a = k.compute_exKxz(self.Z, self.Xmu, self.Xcov)
            b = ek.compute_exKxz(self.Z, self.Xmu, self.Xcov)
            self._assert_pdeq(a, b, k)


if __name__ == '__main__':
    unittest.main()
