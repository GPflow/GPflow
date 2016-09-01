from __future__ import print_function
import GPflow
import numpy as np
import unittest


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
        print('TestGPLVM.optimise')
        m = GPflow.gplvm.GPLVM(self.Y, self.Q)
        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        self.assertTrue(m.compute_log_likelihood() > linit)

    def test_otherkernel(self):
        print('TestGPLVM.test_otherkernel')
        k = GPflow.kernels.PeriodicKernel(self.Q)
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
        k = GPflow.kernels.RBF(Q)
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
        k = GPflow.kernels.RBF(Q)
        m = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean,
                                       X_var=np.ones((self.N, Q)), Y=self.Y, kern=k, M=self.M)
        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        self.assertTrue(m.compute_log_likelihood() > linit)

        # test prediction
        Xtest = self.rng.randn(10, Q)
        mu_f, var_f = m.predict_f(Xtest)
        mu_fFull, var_fFull = m.predict_f_full_cov(Xtest)
        self.assertTrue(np.allclose(mu_fFull, mu_f))
        # check full covariance diagonal
        for i in range(self.D):
            self.assertTrue(np.allclose(var_f[:, i], np.diag(var_fFull[:, :, i])))

#     def test_quadrature(self):
#         print('TestBayesianGPLVM.test_quadrature')
#         ''' This code used PeriodicKernel for which there is are no exact psi statistics computed.
#         So this code results in quadrature being used. Will only work for 1-D case
#         '''
#         k = GPflow.kernels.PeriodicKernel(self.Q)
#         Z = np.linspace(0, 1, self.M)
#         Z = np.expand_dims(Z, self.D)  # inducing points
#         m = GPflow.gplvm.BayesianGPLVM(X_mean=np.zeros((self.N, self.Q)),
#                                        X_var=np.ones((self.N, self.Q)), Y=self.Y, kern=k, Z=Z)
# 
#         linit = m.compute_log_likelihood()
#         m.optimize(maxiter=10)
#         self.assertTrue(m.compute_log_likelihood() > linit)

#     def test_linearSolution(self):
#         # You could implement a standard GPLVM, and show that it recovers PCA when the kernel is linear ->
#         # How to deal with rotations and linear rescalings.
#         pass
# 
#     def test_GPLVM_BGPLVM_Equivalence(self):
#         print('test_GPLVM_BGPLVM_Equivalence')
#         # You could set the variance of the BGPLVM to zero and show that it's the same as the GPLVM
#         # BGPLVM with variance to 0 is same as GPLVM
#         N = 10  # number of data points
#         Q = 1  # latent dimensions
#         M = 5  # inducing points
#         k = GPflow.kernels.RBF(Q)
#         Z = np.linspace(0, 1, M)
#         Z = np.expand_dims(Z, Q)
#         rng = np.random.RandomState(1)
#         Y = rng.randn(N, Q)
#         XInit = rng.rand(N, Q)
#         # use 0 variance for BGPLVM
#         m = GPflow.gplvm.BayesianGPLVM(X_mean=XInit, X_var=np.ones((N, Q)), Y=Y, kern=k, Z=Z)
#         print(m)
#         m.X_var.fixed = True
# 
#         ll = m.compute_log_likelihood()
#         print(ll)
#         m = GPflow.gplvm.BayesianGPLVM(X_mean=XInit, X_var=np.ones((N, Q)), Y=Y, kern=k, Z=Z, X_prior_mean=np.zeros((N, Q)),
#                                        X_prior_var=np.ones((N, Q)))
#         llprior = m.compute_log_likelihood()
#         print(m)
#         print(llprior)
#         assert ll == llprior
# 
#         Z = np.linspace(0, 1, M * 2)
#         Z = np.expand_dims(Z, Q)
#         m = GPflow.gplvm.BayesianGPLVM(X_mean=XInit, X_var=np.ones((N, Q)), Y=Y, kern=k, Z=Z, X_prior_mean=np.zeros((N, Q)),
#                                        X_prior_var=np.ones((N, Q)))
#         llmoreZ = m.compute_log_likelihood()
#         print(llmoreZ)
#         assert llmoreZ > ll
# 
# #         m.optimize()
# #         mGPLVM = GPflow.gplvm.GPLVM(Y=Y, Q=Q, kern=k, XInit=XInit)
# #         mGPLVM.optimize()
# #         assert np.allclose(m.X_mean.value, mGPLVM.X.value)
#         # this does not work - f=    +Infinity!
# 
#     def test_gplvmOptimization(self):
#         print('Run optimisation')
# #         self.m.optimize()


class TestBayesianGPLVMQuadrature(unittest.TestCase):
    def setUp(self):
        # data
        self.N = 20  # number of data points
        self.D = 5  # data dimension
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn(self.N, self.D)
        # model
        self.M = 10  # inducing points

    def testWithPeriodicK(self):
        # test kernel whose Psi statistics not computed
        Q = 1  # latent dimensions
        X_mean = GPflow.gplvm.PCA_reduce(self.Y, Q)
        k = GPflow.kernels.PeriodicKernel(Q)
        m = GPflow.gplvm.BayesianGPLVM(X_mean=X_mean,
                                       X_var=np.ones((self.N, Q)), Y=self.Y, kern=k, M=self.M)
        linit = m.compute_log_likelihood()
        m.optimize(maxiter=2)
        assert(m.compute_log_likelihood() > linit)


if __name__ == "__main__":
    unittest.main()
