# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import GPflow
import unittest


class test_diag_1dim(unittest.TestCase):
    
    def test(self):
        src = np.random.randn(3,4,5)
        src_tf = tf.constant(src)
        
        dest= tf.Session().run(GPflow.tf_hacks.diag_1dim(src_tf))
        ref = np.zeros((3,3,4,5))
        
        for i in range(src.shape[0]):
            ref[i,i, :, :] = src[i,:,:]
        self.assertTrue(np.allclose(dest.shape, ref.shape))
        
        for i in range(ref.shape[0]):
            for j in range(ref.shape[1]):
                self.assertTrue(np.allclose(dest[i,j,:,:], ref[i,j,:,:]))
        
if __name__ == "__main__":
    unittest.main()
