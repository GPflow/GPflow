# -*- coding: utf-8 -*-

import numpy as np
import GPflow
from GPflow.coregionalized.kernels import Linear
from GPflow.coregionalized.labeled_data import LabeledData
import tensorflow as tf
import unittest
        

class test_LinearKernel(unittest.TestCase):
    def test_K(self):
        number_of_tasks = 3
        rank = 2
        w = np.random.randn(number_of_tasks, rank)
        kappa = np.exp(np.random.randn(number_of_tasks))
        base_kern = GPflow.kernels.Constant(input_dim=1)
        base_kern.fixed=True
        model = GPflow.param.Parameterized()
        model.K = Linear(base_kern, number_of_tasks, rank, w, kappa)
        
        index1 = np.random.randint(0, number_of_tasks, 10)
        index2 = np.random.randint(0, number_of_tasks, 11)
        
        # make a reference
        Kx_np = np.zeros((len(index1), len(index1)))
        Kxx_np = np.zeros((len(index1), len(index2)))
        Kdiag_np = np.zeros(len(index1))
        K_value = np.dot(w, np.transpose(w)) + np.diag(kappa)
        for i in range(len(index1)):
            for j in range(len(index1)):
                Kx_np[i, j] = K_value[index1[i], index1[j]]
            for j in range(len(index2)):
                Kxx_np[i, j] = K_value[index1[i],index2[j]]
            Kdiag_np[i] = K_value[index1[i],index1[i]]
        
        X1 = np.random.randn(10,3)        
        X2 = np.random.randn(11,3)        
        model.XL1 = LabeledData((X1, index1), on_shape_change='raise')
        model.XL2 = LabeledData((X2, index2), on_shape_change='raise')

        sess = tf.Session()
        free_array = model.get_free_state()
        model.make_tf_array(free_array)
        feed_dict = model.get_feed_dict()
        sess.run(tf.initialize_all_variables())
        with model.tf_mode():
            Kx_tf = sess.run(model.K.K(model.XL1), feed_dict=feed_dict)
            Kxx_tf = sess.run(model.K.K(model.XL1, model.XL2), feed_dict=feed_dict)
            Kdiag_tf = sess.run(model.K.Kdiag(model.XL1), feed_dict=feed_dict)
        
        self.assertTrue(np.allclose(Kx_np, Kx_tf))
        self.assertTrue(np.allclose(Kxx_np, Kxx_tf))
        self.assertTrue(np.allclose(Kdiag_np, Kdiag_tf))
    
    
if __name__ == "__main__":
    unittest.main()
