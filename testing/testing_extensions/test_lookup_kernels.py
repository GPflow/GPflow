# -*- coding: utf-8 -*-

import numpy as np
from GPflow.extensions.lookup_kernels import LinearCoregionalizedKernel
from GPflow.extensions.index_holder import IndexHolder
import tensorflow as tf
import unittest
        

class test_LinearCoregionalizedKernel(unittest.TestCase):
    def test_K(self):
        number_of_tasks = 3
        rank = 2
        w = np.random.randn(number_of_tasks, rank)
        kappa = np.exp(np.random.randn(number_of_tasks))
        K = LinearCoregionalizedKernel(number_of_tasks, rank, w, kappa)
        K.construct_Kronecker()
        
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

        x1_tf = tf.constant(np.random.randn(len(index1),3), dtype='float64')
        x2_tf = tf.constant(np.random.randn(len(index2),3), dtype='float64')

        K.set_index(IndexHolder(index1), IndexHolder(index2))
        sess = tf.Session()
        free_array = K.get_free_state()
        K.make_tf_array(free_array)
        feed_dict = K.get_feed_dict()
        sess.run(tf.initialize_all_variables())
        with K.tf_mode():
            Kx_tf = sess.run(K.K(x1_tf), feed_dict=feed_dict)
            Kxx_tf = sess.run(K.K(x1_tf, x2_tf), feed_dict=feed_dict)
            Kdiag_tf = sess.run(K.Kdiag(x1_tf), feed_dict=feed_dict)
        
        self.assertTrue(np.allclose(Kx_np, Kx_tf))
        self.assertTrue(np.allclose(Kxx_np, Kxx_tf))
        self.assertTrue(np.allclose(Kdiag_np, Kdiag_tf))
    
    
if __name__ == "__main__":
    unittest.main()
