# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import GPflow
import unittest


class test_gpr(unittest.TestCase):
    def setUp(self):
        #This functions generate data corresponding to two outputs
        f_output1 = lambda x: 4. * np.cos(x/5.) + np.random.rand(x.size) * 2.
        f_output2 = lambda x: 6. * np.cos(x/5.) + np.random.rand(x.size) * 8.
        
        #{X,Y} training set for each output
        X1 = np.random.rand(10); X1=X1*75
        X2 = np.random.rand(11); X2=X2*70 + 30
        Y1 = f_output1(X1)
        Y2 = f_output2(X2)

        X = np.concatenate([X1, X2]).reshape(-1,1)
        Y = np.concatenate([Y1, Y2]).reshape(-1,1)
        # index for training
        index = [0] * len(X1) + [1]*len(X2)        
        
        K_rbf = GPflow.kernels.RBF(1)
        K_rbf.variance.fixed = True
        K= GPflow.coregionalized.kernels.Linear(\
                                K_rbf, number_of_tasks=2, rank=1)
        
        # define a coregionalized model
        self.model = GPflow.coregionalized.gpr.GPR(X, Y, index, kern=K)
        
        
    def test_optimize(self):
        self.model.optimize()
        
    def test_predict(self):
        #{X,Y} training set for each output
        X1 = np.random.rand(12); X1=X1*75
        X2 = np.random.rand(13); X2=X2*70 + 30
        X = np.concatenate([X1, X2]).reshape(-1,1)
        # index for training
        index = [0] * len(X1) + [1]*len(X2)        
        # test predict_f
        fmu, fvar = self.model.predict_f((X, index))
        
    def test_predict_y(self):
        #{X,Y} training set for each output
        X1 = np.random.rand(12); X1=X1*75
        X2 = np.random.rand(13); X2=X2*70 + 30
        X = np.concatenate([X1, X2]).reshape(-1,1)
        # index for training
        index = [0] * len(X1) + [1]*len(X2)        
        # test predict_y
        fmu= self.model.predict_y((X, index))

    def test_predict_density(self):
        pass

# reference
class Constant(GPflow.mean_functions.MeanFunction):
    def __init__(self, c=np.zeros(1)):
        GPflow.mean_functions.MeanFunction.__init__(self)
        self.c = GPflow.param.Param(c.reshape(-1,1))

    def __call__(self, X):
        return tf.matmul(tf.ones_like(X), self.c)
    

class test_gpr_with_mewan(unittest.TestCase):
    def setUp(self):
        #{X,Y} training set for each output
        X1 = np.random.rand(10); X1=X1*75
        X2 = np.random.rand(11); X2=X2*70 + 30
        Y1 = np.cos(X1/5.) + np.random.rand(X1.size) * 2.
        Y2 = np.cos(X2/5.) + np.random.rand(X2.size) * 8.

        self.X = np.concatenate([X1, X2]).reshape(-1,1)
        self.Y = np.concatenate([Y1, Y2]).reshape(-1,1)
        # index for training
        self.label = [0] * len(X1) + [1]*len(X2)        
        
        K_rbf = GPflow.kernels.RBF(1)
        K_rbf.variance.fixed = True
        self.K= GPflow.coregionalized.kernels.Linear(\
                                K_rbf, number_of_tasks=2, rank=1)

    def test_constant(self):        
#        mean_each = [GPflow.mean_functions.Constant(), GPflow.mean_functions.Constant()]
        mean_each = [Constant(), Constant()]
        mean = GPflow.coregionalized.mean_functions.MeanFunction(mean_each)        
        
        # define a coregionalized model
        self.model = GPflow.coregionalized.gpr.GPR(self.X, self.Y, self.label,\
                                                 kern=self.K, mean_function=mean)
        self.model.optimize()

    def test_linear(self):        
        mean_each = [GPflow.mean_functions.Linear(), GPflow.mean_functions.Linear()]
        mean = GPflow.coregionalized.mean_functions.MeanFunction(mean_each)        
        
        # define a coregionalized model
        self.model = GPflow.coregionalized.gpr.GPR(self.X, self.Y, self.label,\
                                                 kern=self.K, mean_function=mean)
        self.model.optimize()

            
        
if __name__ == "__main__":
    unittest.main()
