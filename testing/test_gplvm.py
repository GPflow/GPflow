from __future__ import print_function
import GPflow
import numpy as np
import unittest
from GPflow import gplvm

class TestBayesianGPLVM(unittest.TestCase):
    def setUp(self):
        N = 10 # number of data points
        D = 1  # latent dimensions
        M = 5 # inducings points
        R = 2 # data dimension
        k = GPflow.kernels.RBF(D)
        Z = np.linspace(0,1,M)
        Z = np.expand_dims(Z, D)
        rng = np.random.RandomState(1)
        Y = rng.randn(N,R)
        self.m = GPflow.gplvm.BayesianGPLVM(X_mean = np.zeros((N,D)), 
                    X_var=np.ones((N,D)), Y=Y, kern=k, Z=Z)

    def test_linearSolution(self):
        # You could implement a standard GPLVM, and show that it recovers PCA when the kernel is linear -> 
        # How to deal with rotations and linear rescalings.
        pass

    def test_GPLVM_BGPLVM_Equivalence(self):
        # You could set the variance of the BGPLVM to zero and show that it's the same as the GPLVM
        # BGPLVM with variance to 0 is same as GPLVM
        N = 10  # number of data points
        Q = 1  # latent dimensions
        M = 5  # inducing points
        D = 2  # data dimension
        k = GPflow.kernels.RBF(Q)
        Z = np.linspace(0, 1, M)
        Z = np.expand_dims(Z, Q)
        rng = np.random.RandomState(1)
        Y = rng.randn(N, Q)
        XInit = rng.rand(N, Q)
        # use 0 variance for BGPLVM
        m = GPflow.gplvm.BayesianGPLVM(X_mean=XInit, X_var=np.ones((N, Q)), Y=Y, kern=k, Z=Z)
        print(m)
        m.X_var.fixed = True

        ll = m.compute_log_likelihood()
        print(ll)
        m = GPflow.gplvm.BayesianGPLVM(X_mean=XInit, X_var=np.ones((N, Q)), Y=Y, kern=k, Z=Z, X_prior_mean=np.zeros((N,Q)), X_prior_var = np.ones((N,Q)))
        llprior = m.compute_log_likelihood()
        print(m) 
        print(llprior)
        assert ll == llprior
 
        Z = np.linspace(0, 1, M*2)
        Z = np.expand_dims(Z, Q)
        m = GPflow.gplvm.BayesianGPLVM(X_mean=XInit, X_var=np.ones((N, Q)), Y=Y, kern=k, Z=Z, X_prior_mean=np.zeros((N,Q)), X_prior_var = np.ones((N,Q)))
        llmoreZ = m.compute_log_likelihood()
        print(llmoreZ)
        assert llmoreZ > ll
        
#         m.optimize()
#         mGPLVM = GPflow.gplvm.GPLVM(Y=Y, Q=Q, kern=k, XInit=XInit)
#         mGPLVM.optimize()
#         assert np.allclose(m.X_mean.value, mGPLVM.X.value)
        # this does not work - f=    +Infinity!

    def test_gplvmOptimization(self):
        print('Run optimisation')
#         self.m.optimize()
    


if __name__ == "__main__":
    unittest.main()

