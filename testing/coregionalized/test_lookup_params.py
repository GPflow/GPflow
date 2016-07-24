# -*- coding: utf-8 -*-

import numpy as np
import GPflow
from GPflow.coregionalized.lookup_params import LookupParam, LookupDictData
from GPflow.coregionalized.labeled_data import LabelHolder
import tensorflow as tf
import unittest
        

class test_lookup_param(unittest.TestCase):
    
    def test_call(self):
        p_original = np.random.randn(3,4,5)
        p = LookupParam(p_original)
        index_i = np.random.randint(0,3, 10).reshape(-1,1)
        index_j = np.random.randint(0,4, 11).reshape(-1,1)
        for i in index_i:
            self.assertTrue(
             np.allclose(p(i), p_original[i]))
            for j in index_j:
                self.assertTrue(np.allclose(p(i,j), p_original[i,j]))
                
    def test_shape_update(self):
        p_original = np.random.randn(3,4,5)
        p = LookupParam(p_original)
        # update table
        p_new = np.random.randn(4,3,2)        
        p.table = GPflow.param.Param(p_new)
        self.assertTrue(np.allclose(p.shape, p_new.shape))
   
    def test_diag(self):
        p_original = np.random.randn(3)
        p = LookupParam(p_original)
        index_i = np.random.randint(0,3,10)
        index_j = np.random.randint(0,3,12)
        for i in index_i:
            for j in index_j:
                if i==j:
                    self.assertTrue(np.allclose(p.diag(i,j), p_original[i]))
                else:
                    self.assertTrue(np.allclose(p.diag(i,j), 0.0))
    
    def test_call_tf(self):
        """ for tf_mode"""
        p_original = np.random.randn(3,4,5)
        model = GPflow.param.Parameterized()
        model.p = LookupParam(p_original)

        index_i = np.random.randint(0, 3, 10)
        index_j = np.random.randint(0, 4, 11)
        
        model.labelholder_i = LabelHolder(index_i)
        model.labelholder_j = LabelHolder(index_j)
        
        value_i  = np.zeros((10, p_original.shape[1], p_original.shape[2]))
        value_ij = np.zeros((10, 11, p_original.shape[2]))
        
        # for reference
        for i in range(len(index_i)):
            value_i[i] = p_original[index_i[i]]
            for j in range(len(index_j)):
                value_ij[i,j,:] = p_original[index_i[i],index_j[j]]

        sess = tf.Session()
        feed_dict = model.labelholder_i.get_feed_dict()
        feed_dict.update(model.labelholder_j.get_feed_dict())
        
        free_array = model.p.get_free_state()
        model.p.make_tf_array(free_array)
        with model.tf_mode():
            sess.run(tf.initialize_all_variables())
            value_i_tf = sess.run(model.p(model.labelholder_i.label), feed_dict=feed_dict)
            value_ij_tf = sess.run(model.p(model.labelholder_i.label, model.labelholder_j.label), feed_dict=feed_dict)

        self.assertTrue(np.allclose(value_i, value_i_tf))
        self.assertTrue(np.allclose(value_ij, value_ij_tf))
        sess.close()

        
    def test_diag_tf(self):
        p_original = np.random.randn(3)
        model = GPflow.param.Parameterized()
        model.p = LookupParam(p_original)

        index_i = np.random.randint(0, 3, 3)
        index_j = np.random.randint(0, 3, 4)
        model.labelholder_i = LabelHolder(index_i)
        model.labelholder_j = LabelHolder(index_j)

        value_ij  = np.zeros((3, 4))
        for i in range(len(index_i)):
            for j in range(len(index_j)):
                value_ij[i,j] = model.p.diag(index_i[i],index_j[j])

        sess = tf.Session()        
        feed_dict = model.labelholder_i.get_feed_dict()
        feed_dict.update(model.labelholder_j.get_feed_dict())
        
        free_array = model.p.get_free_state()
        model.p.make_tf_array(free_array)
        with model.tf_mode():
            sess.run(tf.initialize_all_variables())
            value_ij_tf = sess.run(\
                model.p.diag(model.labelholder_i.label, \
                             model.labelholder_j.label), feed_dict = feed_dict)
        
        self.assertTrue(np.allclose(value_ij, value_ij_tf))                   
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