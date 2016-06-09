from __future__ import print_function
import GPflow
from GPflow import gplvm
import tensorflow as tf
import numpy as np
import unittest


class TestGPLVM(unittest.TestCase):
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

    def test_gplvmOptimization(self):
        self.m.optimize()
    


if __name__ == "__main__":
    unittest.main()

