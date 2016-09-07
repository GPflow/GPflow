import unittest
import numpy as np
import numpy.random as rnd
import GPflow.ekernels as kernels
import GPflow.etransforms


class TestKernExpMC(unittest.TestCase):
    """
    Perform a Monte Carlo estimate of the expectation and compare to the calculated version.
    """

    def setUp(self):
        self.D = 2
        self.N = 4
        self.samples = 1000000
        self.Xmu = rnd.rand(self.N, self.D)
        self.Z = rnd.rand(3, self.D)

        unconstrained = rnd.randn(self.N, 2 * self.D, self.D)
        t = GPflow.etransforms.TriDiagonalBlockRep()
        self.Xcov = t.forward(unconstrained)

        self.k = kernels.RBF(self.D, ARD=True)
        self.k.lengthscales = rnd.rand(self.D) + [0.5, 1.5][:self.D]
        self.k.variance = 0.3 + rnd.rand()

        self.Xs = t.sample(self.Xcov, self.samples).transpose([1, 2, 0]) + self.Xmu[:, :, None]
        self.Ksamples = self.k.compute_K(self.Xs.transpose([2, 0, 1]).reshape(-1, self.D), self.Z) \
            .reshape(self.samples, self.Xmu.shape[0], -1)  # SxNxM

    def test_eKzxKxz_monte_carlo(self):
        # Calculation
        m = self.k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov[0, :, :, :])

        # Monte Carlo estimate
        outer = np.einsum('sid,sie->side', self.Ksamples, self.Ksamples)
        outer_mean = np.mean(outer, 0)

        maxpd = np.max(np.abs((outer_mean - m) / m * 100))  # Evaluation
        self.assertTrue(maxpd < 0.5, msg="Difference is %f." % maxpd)

    def test_eKdiag_monte_carlo(self):
        m = self.k.compute_eKdiag(self.Xmu, self.Xcov[0, :, :, :])

        # MC estimate
        Ksamples = self.k.compute_Kdiag(self.Xs.transpose(2, 0, 1).reshape(-1, self.D)) \
            .reshape(self.samples, self.Xmu.shape[0])
        est = np.mean(Ksamples, 0)
        maxpd = np.max(np.abs(est - m) / m * 100)
        self.assertTrue(maxpd < 10 ** -3, msg="Difference is %f." % maxpd)

    def test_exKxz(self):
        """
        This test seems to have high variance in the monte carlo estimate. Would be good to test this against a
        reference implementation as well.
        :return:
        """
        m = self.k.compute_exKxz(self.Z, self.Xmu, self.Xcov)  # NxMxD

        # Monte Carlo estimate
        outer_mean = np.mean(np.einsum('snm,nds->snmd', self.Ksamples[:, :-1, :], self.Xs[1:, :]), 0)

        pd = (outer_mean - m) / m * 100
        maxpd = np.max(np.abs(pd))  # Evaluation
        self.assertTrue(maxpd < 5.0, msg="Difference is %f." % maxpd)


class TestKernExpDelta(unittest.TestCase):
    """
    Check whether the normal kernel matrix is recovered if a delta distribution is used.
    """

    def setUp(self):
        self.D = 2
        self.Xmu = rnd.rand(10, self.D)
        self.Z = rnd.rand(4, self.D)
        self.Xcov = np.zeros((self.Xmu.shape[0], self.D, self.D))
        self.Xcovc = np.zeros((self.Xmu.shape[0], self.D, self.D))
        self.k = kernels.RBF(self.D, ARD=True)
        self.k.lengthscales = rnd.rand(2) + [0.5, 1.5]
        self.k.variance = 0.3 + rnd.rand()

    def test_eKzxKxz(self):
        # Calculation
        psi2 = self.k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov)
        kernmat = self.k.compute_K(self.Z, self.Xmu)  # MxN
        kernouter = np.einsum('in,jn->nij', kernmat, kernmat)
        self.assertTrue(np.allclose(kernouter, psi2))

    def test_eKdiag(self):
        kdiag = self.k.compute_eKdiag(self.Xmu, self.Xcov)
        orig = self.k.compute_Kdiag(self.Xmu)
        self.assertTrue(np.allclose(orig, kdiag))

    def test_exKxz(self):
        covall = np.array([self.Xcov, self.Xcovc])
        exKxz = self.k.compute_exKxz(self.Z, self.Xmu, covall)
        Kxz = self.k.compute_K(self.Xmu[:-1, :], self.Z)  # NxM
        xKxz = np.einsum('nm,nd->nmd', Kxz, self.Xmu[1:, :])

        self.assertTrue(np.allclose(xKxz, exKxz))


if __name__ == '__main__':
    unittest.main()
