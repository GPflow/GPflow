import GPflow
from GPflow.coregionalized.labeled_data import LabeledData
import tensorflow as tf
import numpy as np
import unittest

        
        
class test_labeled_data(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_set_data(self):
        data = np.array(np.random.randn(6,2))
        label = [1, 0, 2, 1, 0, 0]
        # answers
        permutation = [[1, 4, 5], [0, 3], [2]]
        inv_perm = [3, 0, 5, 4, 1, 2]
        labeled_data = LabeledData((data, label), on_shape_change='pass')
        
        # test permutation
        for i in range(len(permutation)):
            self.assertTrue(np.equal(permutation[i], labeled_data._permutation[i].value).all())
        self.assertTrue(np.equal(inv_perm, labeled_data._inv_perm.value).all())            
        
        # another case
        data = np.array(np.random.randn(3,3))
        index = [0, 1, 2]
        permutation = [[0,], [1,], [2,]]
        inv_perm = [0, 1, 2]
        labeled_data.set_data((data, index))
        for i in range(len(permutation)):
            self.assertTrue(np.equal(permutation[i], labeled_data._permutation[i].value).all())
        self.assertTrue(np.equal(inv_perm, labeled_data._inv_perm.value).all())            

        # Invalid index.
        # It should raise an exception  
        index = [0, 1, 10]
        self.assertRaises(IndexError, lambda: labeled_data.set_data((data, index)))
        # It should raise an exception  
        index = [-1, 1, 2]
        self.assertRaises(IndexError, lambda: labeled_data.set_data((data, index)))
        
                
    def test_split_tf(self):
        data = np.array(np.random.randn(6,2))
        # for tf_mode
        label = [1, 0, 2, 1, 0, 0]
        labeled_data = LabeledData((data, label), on_shape_change='pass')
        
        ary = np.array([0., 10., 2., 30., 400., 50.], dtype=np.float64).reshape(-1,1)
        ref_holder = GPflow.data_holders.DictData(ary, on_shape_change='pass')
        ref_split = [[[10.], [400.], [50.]], [[0.], [30.]], [[2.]]]

        model = GPflow.param.Parameterized()
        model.labeled_data = labeled_data
        model.ref_holder = ref_holder
        
        sess = tf.Session()
        free_array = model.get_free_state()
        model.make_tf_array(free_array)
        sess.run(tf.initialize_all_variables())
        feed_dict = model.get_feed_dict()
        
        with model.tf_mode():
            split_tf = sess.run(model.labeled_data.split(model.ref_holder), \
                                    feed_dict=feed_dict)
        
        for i in range(len(split_tf)):
            self.assertTrue(np.allclose(split_tf[i], np.array(ref_split[i])))
        
        sess.close()

    def test_restore_tf(self):
        data = np.array(np.random.randn(6,2))
        # for tf_mode
        label = [1, 0, 2, 1, 0, 0]
        labeled_data = LabeledData((data, label), on_shape_change='pass')
        
        ary = np.array([0., 10., 2., 30., 400., 50.], dtype=np.float64).reshape(-1,1)
        ref_holder = GPflow.data_holders.DictData(ary, on_shape_change='pass')

        model = GPflow.param.Parameterized()
        model.labeled_data = labeled_data
        model.ref_holder = ref_holder
        
        sess = tf.Session()
        free_array = model.get_free_state()
        model.make_tf_array(free_array)
        sess.run(tf.initialize_all_variables())
        feed_dict = model.get_feed_dict()
        
        with model.tf_mode():
            split = model.labeled_data.split(model.ref_holder)
            restored = sess.run(model.labeled_data.restore(split), \
                                    feed_dict=feed_dict)
        
        self.assertTrue(np.allclose(ary, restored))
        
        sess.close()
    
if __name__ == "__main__":
    unittest.main()
