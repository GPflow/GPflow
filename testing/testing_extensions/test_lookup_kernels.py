# -*- coding: utf-8 -*-

import numpy as np
from ..GPflow_extensions.lookup_kernels import LookupKern, LinearCoregionalizedKernel
import tensorflow as tf
import unittest

class test_lookup_kern(unittest.TestCase):
    def test_init(self):
        index_dim = 2
        K = LookupKern(index_dim)
        with self.assertRaises(TypeError):
            K_raise = LookupKern([2,1])            
    
    def test_to_index(self):
        index_dim = 2
        K = LookupKern(input_dim=3, index_dim=index_dim)
        x = np.random.randn(10, 9)
        # test with np.array
        self.assertTrue(np.equal(K._to_index(x), np.squeeze(x[:,index_dim]).astype(int)).all())
        
    def test_to_index_tf(self):
        index_dim = 2
        # test for tensor
        x = np.random.randn(10, 9)
        x_tf = tf.Variable(x)
        
        K = LookupKern(input_dim=3, index_dim=index_dim)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        with K.tf_mode():
            index = sess.run(K._to_index(x_tf))

        # test with np.array
        self.assertTrue(np.equal(index, np.squeeze(x[:,index_dim]).astype(int)).all())
        sess.close()
        

class test_LinearCoregionalizedKernel(unittest.TestCase):
    def test_K(self):
        index_dim = 2
        number_of_tasks = 3
        rank = 2
        w = np.random.randn(number_of_tasks, rank)
        kappa = np.exp(np.random.randn(number_of_tasks))
        K = LinearCoregionalizedKernel(index_dim, number_of_tasks, rank, w, kappa)
        
        x = []
        for i in range(10):
            x.append([np.random.randn(1),np.random.randn(1), \
                    np.random.randint(0, number_of_tasks),np.random.randn(1)])
        x1 = np.array(x)
        
        x.append([np.random.randn(1),np.random.randn(1), \
                    np.random.randint(0, number_of_tasks),np.random.randn(1)])
        x2 = np.array(x)

        # make a reference
        Kx_np = np.zeros((len(x1), len(x1)))
        Kxx_np = np.zeros((len(x1), len(x2)))
        Kdiag_np = np.zeros((len(x1), len(x1)))
        K_value = np.dot(w, np.transpose(w)) + np.diag(kappa)
        for i in range(len(x1)):
            for j in range(len(x1)):
                Kx_np[i, j] = K_value[int(x1[i, index_dim]),int(x1[j, index_dim])]
            for j in range(len(x2)):
                Kxx_np[i, j] = K_value[int(x1[i, index_dim]),int(x2[j, index_dim])]
            Kdiag_np[i,i] = K_value[int(x1[i, index_dim]),int(x1[i, index_dim])]

        x1_tf = tf.constant(x1, dtype='float64')
        x2_tf = tf.constant(x2, dtype='float64')
        
        sess = tf.Session()
        free_array = K.get_free_state()
        K.make_tf_array(free_array)
        sess.run(tf.initialize_all_variables())
        with K.tf_mode():
            Kx_tf = sess.run(K.K(x1_tf))
            Kxx_tf = sess.run(K.K(x1_tf, x2_tf))
            Kdiag_tf = sess.run(K.Kdiag(x1_tf))
        
    
        self.assertTrue(np.allclose(Kx_np, Kx_tf))
        self.assertTrue(np.allclose(Kxx_np, Kxx_tf))
        self.assertTrue(np.allclose(Kdiag_np, Kdiag_tf))
    
    
