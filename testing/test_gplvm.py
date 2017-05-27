from __future__ import print_function
import GPflow
import numpy as np
import unittest
from GPflow import ekernels
from GPflow import kernels
np.random.seed(0)


class TestGPLVM(unittest.TestCase):
    def setUp(self):
        # data
        self.N = 20  # number of data points
        D = 5  # data dimension
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn(self.N, D)
        # model
        self.Q = 2  # latent dimensions

    def test_optimise(self):
        m = GPflow.gplvm.GPLVM(self.Y, self.Q)
        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        self.assertTrue(m.compute_log_likelihood() > linit)

    def test_otherkernel(self):
        k = kernels.PeriodicKernel(self.Q)
        XInit = self.rng.rand(self.N, self.Q)
        m = GPflow.gplvm.GPLVM(self.Y, self.Q, XInit, k)
        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        self.assertTrue(m.compute_log_likelihood() > linit)


class TestBayesianGPLVM(unittest.TestCase):
    def setUp(self):
        # data
        self.N = 20  # number of data points
        self.D = 5  # data dimension
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn(self.N, self.D)
        # model
        self.M = 10  # inducing points

    def test_1d(self):
        Q = 1  # latent dimensions
        k = ekernels.RBF(Q)
        Z = np.linspace(0, 1, self.M)
        Z = np.expand_dims(Z, Q)  # inducing points
        m = GPflow.gplvm.BayesianGPLVM(X_mean=np.zeros((self.N, Q)),
                                       X_var=np.ones((self.N, Q)), Y=self.Y, kern=k, M=self.M, Z=Z)
        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        self.assertTrue(m.compute_log_likelihood() > linit)

    def test_2d(self):
        # test default Z on 2_D example
        Q = 2  # latent dimensions
        X_mean = GPflow.gplvm.PCA_reduce(self.Y, Q)
        k = ekernels.RBF(Q, ARD=False)
        m = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean,
                                       X_var=np.ones((self.N, Q)), Y=self.Y, kern=k, M=self.M)
        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        self.assertTrue(m.compute_log_likelihood() > linit)

        # test prediction
        Xtest = self.rng.randn(10, Q)
        mu_f, var_f = m.predict_f(Xtest)
        mu_fFull, var_fFull = m.predict_f(Xtest, full_cov=True)
        self.assertTrue(np.allclose(mu_fFull, mu_f))
        # check full covariance diagonal
        for i in range(self.D):
            self.assertTrue(np.allclose(var_f[:, i], np.diag(var_fFull[:, :, i])))

    def test_kernelsActiveDims(self):
        ''' Test sum and product compositional kernels '''
        Q = 2  # latent dimensions
        X_mean = GPflow.gplvm.PCA_reduce(self.Y, Q)
        kernsQuadratu = [kernels.RBF(1, active_dims=[0])+kernels.Linear(1, active_dims=[1]),
                         kernels.RBF(1, active_dims=[0])+kernels.PeriodicKernel(1, active_dims=[1]),
                         kernels.RBF(1, active_dims=[0])*kernels.Linear(1, active_dims=[1]),
                         kernels.RBF(Q)+kernels.Linear(Q)]  # non-overlapping
        kernsAnalytic = [ekernels.Add([ekernels.RBF(1, active_dims=[0]), ekernels.Linear(1, active_dims=[1])]),
                         ekernels.Add([ekernels.RBF(1, active_dims=[0]), kernels.PeriodicKernel(1, active_dims=[1])]),
                         ekernels.Prod([ekernels.RBF(1, active_dims=[0]), ekernels.Linear(1, active_dims=[1])]),
                         ekernels.Add([ekernels.RBF(Q), ekernels.Linear(Q)])]
        fOnSeparateDims = [True, True, True, False]
        Z = np.random.permutation(X_mean.copy())[:self.M]
        # Also test default N(0,1) is used
        X_prior_mean = np.zeros((self.N, Q))
        X_prior_var = np.ones((self.N, Q))
        Xtest = self.rng.randn(10, Q)
        for kq, ka, sepDims in zip(kernsQuadratu, kernsAnalytic, fOnSeparateDims):
            kq.num_gauss_hermite_points = 20  # speed up quadratic for tests
            ka.kern_list[0].num_gauss_hermite_points = 0  # RBF should throw error if quadrature is used
            if(sepDims):
                self.assertTrue(ka.on_separate_dimensions, 'analytic kernel must not use quadrature')
            mq = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean, X_var=np.ones((self.N, Q)), Y=self.Y,
                                            kern=kq, M=self.M, Z=Z, X_prior_mean=X_prior_mean, X_prior_var=X_prior_var)
            ma = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean, X_var=np.ones((self.N, Q)), Y=self.Y,
                                            kern=ka, M=self.M, Z=Z)
            mq._compile()
            ma._compile()
            ql = mq.compute_log_likelihood()
            al = ma.compute_log_likelihood()
            self.assertTrue(np.allclose(ql, al, atol=1e-2), 'Likelihood not equal %f<>%f' % (ql, al))
            mu_f_a, var_f_a = ma.predict_f(Xtest)
            mu_f_q, var_f_q = mq.predict_f(Xtest)
            self.assertTrue(np.allclose(mu_f_a, mu_f_q, atol=1e-4), ('Posterior means different', mu_f_a-mu_f_q))
            self.assertTrue(np.allclose(mu_f_a, mu_f_q, atol=1e-4), ('Posterior vars different', var_f_a-var_f_q))


if __name__ == "__main__":
    unittest.main()
