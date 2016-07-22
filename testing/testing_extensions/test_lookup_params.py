# -*- coding: utf-8 -*-

import numpy as np
import GPflow
from GPflow.extensions.lookup_params import LookupParam, LookupDictData
from GPflow.extensions.index_holder import IndexHolder
import tensorflow as tf
import unittest
        

class test_lookup_param(unittest.TestCase):
    def setUp(self):
        self.p_original = np.random.randn(3,4,5)
        self.p = LookupParam(self.p_original)
    
    def test_call(self):
        index_i = np.random.randint(0,3, 10).reshape(-1,1)
        index_j = np.random.randint(0,4, 11).reshape(-1,1)
        for i in index_i:
            self.assertTrue(
             np.allclose(self.p(i), self.p_original[i]))
            for j in index_j:
                self.assertTrue(np.allclose(self.p(i,j),
                                    self.p_original[i,j]))
    
    def test_diag(self):
        index_i = np.random.randint(0,3, 10)
        index_j = np.random.randint(0,3, 10)
        for i in index_i:
            for j in index_j:
                if i == j:
                    self.assertTrue(np.allclose(self.p.diag(i,j), self.p_original[i]))
                else:
                    self.assertTrue(np.allclose(self.p.diag(i, j), np.zeros((4,5))))
                
    
    def test_call_tf(self):
        """ for tf_mode"""
        model = GPflow.param.Parameterized()
        model.p = LookupParam(self.p_original)

        index_i = np.random.randint(0, 3, 10)
        index_j = np.random.randint(0, 4, 11)
        
        model.indexholder_i = IndexHolder(index_i)
        model.indexholder_j = IndexHolder(index_j)
        
        value_i  = np.zeros((10, self.p_original.shape[1], self.p_original.shape[2]))
        value_ij = np.zeros((10, 11, self.p_original.shape[2]))
        
        # for reference
        for i in range(len(index_i)):
            value_i[i] = self.p_original[index_i[i]]
            for j in range(len(index_j)):
                value_ij[i,j,:] = self.p_original[index_i[i],index_j[j]]

        sess = tf.Session()
        feed_dict = model.indexholder_i.get_feed_dict()
        feed_dict.update(model.indexholder_j.get_feed_dict())
        
        free_array = model.p.get_free_state()
        model.p.make_tf_array(free_array)
        with model.tf_mode():
            sess.run(tf.initialize_all_variables())
            value_i_tf = sess.run(model.p(model.indexholder_i.index), feed_dict=feed_dict)
            value_ij_tf = sess.run(model.p(model.indexholder_i.index, model.indexholder_j.index), feed_dict=feed_dict)

        self.assertTrue(np.allclose(value_i, value_i_tf))
        self.assertTrue(np.allclose(value_ij, value_ij_tf))
        sess.close()

        
    def test_diag_tf(self):
        model = GPflow.param.Parameterized()
        model.p = LookupParam(self.p_original)

        index_i = np.random.randint(0, 3, 10)
        index_j = np.random.randint(0, 3, 10)
        
        model.indexholder_i = IndexHolder(index_i)
        model.indexholder_j = IndexHolder(index_j)

        value_i  = np.zeros((10, 10, self.p_original.shape[1], self.p_original.shape[2]))
        for i in range(len(index_i)):
            for j in range(len(index_j)):
                value_i[i,j] = self.p.diag(index_i[i], index_j[j])

        sess = tf.Session()        
        feed_dict = model.indexholder_i.get_feed_dict()
        feed_dict.update(model.indexholder_j.get_feed_dict())
        
        free_array = model.p.get_free_state()
        model.p.make_tf_array(free_array)
        with model.tf_mode():
            sess.run(tf.initialize_all_variables())
            value_i_tf = sess.run(\
                model.p.diag(model.indexholder_i.index, model.indexholder_j.index), \
                feed_dict = feed_dict)
        
        self.assertTrue(np.allclose(value_i, value_i_tf))                   
        sess.close()        
        
        
class test_lookup_data(test_lookup_param):
    """   
    The same test for LookupDictData
    """    
    def setUp(self):
        self.p_original = np.random.randn(3,4,5)
        self.p = LookupDictData(self.p_original)


if __name__ == "__main__":
    unittest.main()