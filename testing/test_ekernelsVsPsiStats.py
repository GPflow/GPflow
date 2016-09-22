import numpy as np
import numpy.random as rnd
import GPflow.ekernels as kernels
import GPflow
import tensorflow as tf
import unittest


def is_one_dimensional(kern):
    with tf.Session() as sess:
        dims = sess.run(kern.active_dims)
    return len(dims) == 1


def pad_inputs(kern, X):
    """
    prepend extra columns to X that will be sliced away by the kernel
    """
    return tf.concat(1, [tf.zeros(tf.pack([tf.shape(X)[0], kern.active_dims[0]]), tf.float64), X])


def one_dimensional_psi_stats(Z, kern, mu, S, numpoints=2):
    """
    This function computes the psi-statistics for an arbitrary kernel with only one input dimension.
    """
    assert is_one_dimensional(kern)

    # only use the active dimensions.
    mu, S = kern._slice(mu, S)
    Z, _ = kern._slice(Z, None)

    # compute a grid over which to compute approximate the integral
    gh_x, gh_w = np.polynomial.hermite.hermgauss(numpoints)
    gh_w /= np.sqrt(np.pi)
    X = gh_x * tf.sqrt(2.0 * S) + mu

    p0 = [kern.Kdiag(pad_inputs(kern, X[:, i:i+1]))*wi for i, wi in enumerate(gh_w)]
    psi0 = tf.reduce_sum(tf.pack(p0), 0)  # vector of size N

    # psi1
    KXZ = [kern.K(pad_inputs(kern, X[:, i:i+1]), Z) for i in range(numpoints)]
    p1 = [KXZ_i*wi for KXZ_i, wi in zip(KXZ, gh_w)]
    psi1 = tf.pack(p1)

    # psi2
    p2 = [tf.matmul(tf.transpose(KXZ_i), KXZ_i)*wi for KXZ_i, wi in zip(KXZ, gh_w)]
    psi2 = tf.pack(p2)

    return psi0, psi1, psi2


class PsiComputer(GPflow.param.Parameterized):
    def __init__(self, kern):
        GPflow.param.Parameterized.__init__(self)
        self.kern = kern

    @GPflow.param.AutoFlow((tf.float64,), (tf.float64,), (tf.float64,))
    def psi0(self, mu, S, Z):
        return GPflow.kernel_expectations.build_psi_stats(Z, self.kern, mu, S)[0]

    @GPflow.param.AutoFlow((tf.float64,), (tf.float64,), (tf.float64,))
    def psi1(self, mu, S, Z):
        return GPflow.kernel_expectations.build_psi_stats(Z, self.kern, mu, S)[1]

    @GPflow.param.AutoFlow((tf.float64,), (tf.float64,), (tf.float64,))
    def psi2(self, mu, S, Z):
        return GPflow.kernel_expectations.build_psi_stats(Z, self.kern, mu, S)[2]

    @GPflow.param.AutoFlow((tf.float64,), (tf.float64,), (tf.float64,))
    def psiQuad(self, mu, S, Z):
        return one_dimensional_psi_stats(Z, self.kern, mu, S)


class TestRBFKernelExpectations(unittest.TestCase):
    """
    Test RBF kernel expectations via both quadrature and known analytic results.
    """

    def setUp(self):
        D = 2  # number of latent dimensions
        M = 5  # number of latent points
        N = 12  # number of data points
        self.Xmu = rnd.rand(N, D)
        self.Z = rnd.rand(M, D)
        self.Xcov = np.zeros((self.Xmu.shape[0], D, D))
        self.D = D

    def test_kern(self):
        k = kernels.RBF(self.D, ARD=True)
        k.lengthscales = rnd.rand(self.D) + 1.5
        k.variance = 0.3 + rnd.rand()
        #self.quad(k)
        self.psi0(k)
        self.psi1(k)
        self.psi2(k)

    def quad(self, k):
        # Check via quadrature
        p0, p1, p2 = PsiComputer(k).psiQuad(np.atleast_2d(self.Xmu), self.Xcov[:, :, 0], self.Z)
        # ekernels code
        kdiag = k.compute_eKdiag(self.Xmu, self.Xcov)
        # psi2 = k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov) TO DO

        self.assertTrue(np.allclose(kdiag, p0))
        # self.assertTrue(np.allclose(kdiag, p1))  # TODO check psi1 stats
        # self.assertTrue(np.allclose(psi2, p2))

    def psi0(self, k):
        # Check via analytic psi code
        psi0_ke = PsiComputer(k).psi0(np.atleast_2d(self.Xmu), np.atleast_2d(self.Xcov[:, 0, 0]), self.Z)
        kdiag = k.compute_eKdiag(self.Xmu, self.Xcov)
        self.assertTrue(np.allclose(kdiag.sum(), psi0_ke))

    def psi1(self, k):
        psi1_ke = PsiComputer(k).psi1(np.atleast_2d(self.Xmu), self.Xcov[:, :, 0], self.Z)
        psi1 = k.compute_eKxz(self.Z, self.Xmu, self.Xcov)
        self.assertTrue(np.allclose(psi1_ke, psi1))

    def psi2(self, k):
        psi2_ke = PsiComputer(k).psi2(np.atleast_2d(self.Xmu), np.atleast_2d(self.Xcov[:, 0, 0]), self.Z)
        psi2 = k.compute_eKzxKxz(self.Z, self.Xmu, self.Xcov)
        self.assertTrue(np.allclose(psi2_ke, psi2.sum(0)))

if __name__ == '__main__':
    unittest.main()
