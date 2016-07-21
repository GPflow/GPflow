import GPflow
from GPflow.extensions.index_holder import IndexHolder
import tensorflow as tf
import numpy as np
import unittest

class test_index_holder(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_set_index(self):
        index = [1, 0, 2, 1, 0, 0]
        # answers
        permutation = [[1, 4, 5], [0, 3], [2]]
        inv_perm = [3, 0, 5, 4, 1, 2]
        index_holder = IndexHolder(np.array(index, dtype=np.int32), on_shape_change='pass')
        # test permutation
        for i in range(len(permutation)):
            self.assertTrue(np.equal(permutation[i], index_holder._permutation[i].value).all())
        self.assertTrue(np.equal(inv_perm, index_holder._inv_perm.value).all())            
        
        # another case
        index = [0, 1, 2]
        permutation = [[0,], [1,], [2,]]
        inv_perm = [0, 1, 2]
        index_holder.set_index(index)
        for i in range(len(permutation)):
            self.assertTrue(np.equal(permutation[i], index_holder._permutation[i].value).all())
        self.assertTrue(np.equal(inv_perm, index_holder._inv_perm.value).all())            

        # different num_index
        index_holder.set_index(index, num_index=4)
        permutation = [[0,], [1,], [2,], []]
        for i in range(len(permutation)):
            self.assertTrue(np.equal(permutation[i], index_holder._permutation[i].value).all())
        self.assertTrue(np.equal(inv_perm, index_holder._inv_perm.value).all())            

        
    def test_split(self):
        index = [1, 0, 2, 1, 0, 0]
        # answers
        #ref_permutation = [[1, 4, 5], [0, 3], [2]]
        #ref_inv_perm = [3, 0, 5, 4, 1, 2]
        index_holder = IndexHolder(np.array(index, dtype=np.int32), on_shape_change='pass')
        
        ary = np.array([0., 10., 2., 30., 400., 50.], dtype=np.float64).reshape(-1,1)
        ref_holder = GPflow.param.DataHolder(ary, on_shape_change='pass')

        split = index_holder.split(ref_holder)
        ref_split = [[[10.], [400.], [50.]], [[0.], [30.]], [[2.]]]
        # for not tf_mode
        for i in range(len(split)):
            self.assertTrue(np.allclose(split[i], np.array(ref_split[i])))
        
    def test_split_tf(self):
        # for tf_mode
        index = [1, 0, 2, 1, 0, 0]
        index_holder = IndexHolder(np.array(index, dtype=np.int32), on_shape_change='pass')
        ary = np.array([0., 10., 2., 30., 400., 50.], dtype=np.float64).reshape(-1,1)
        ref_holder = GPflow.param.DataHolder(ary, on_shape_change='pass')
        ref_split = [[[10.], [400.], [50.]], [[0.], [30.]], [[2.]]]

        model = GPflow.param.Parameterized()
        model.index_holder = index_holder
        model.ref_holder = ref_holder
        
        sess = tf.Session()
        free_array = model.get_free_state()
        model.make_tf_array(free_array)
        sess.run(tf.initialize_all_variables())
        feed_dict = model.get_feed_dict()
        
        with model.tf_mode():
            split_tf = sess.run(model.index_holder.split(model.ref_holder), \
                                    feed_dict=feed_dict)
        
        for i in range(len(split_tf)):
            self.assertTrue(np.allclose(split_tf[i], np.array(ref_split[i])))
        
        sess.close()

    def test_restore(self):
        index = [1, 0, 2, 1, 0, 0]
        # answers
        #ref_permutation = [[1, 4, 5], [0, 3], [2]]
        #ref_inv_perm = [3, 0, 5, 4, 1, 2]
        index_holder = IndexHolder(np.array(index, dtype=np.int32), on_shape_change='pass')
        
        ary = np.array([0., 10., 2., 30., 400., 50.], dtype=np.float64).reshape(-1,1)
        ref_holder = GPflow.param.DataHolder(ary, on_shape_change='pass')

        split = index_holder.split(ref_holder)
        ary_restored = index_holder.restore(split)
        self.assertTrue(np.allclose(ary, ary_restored))

    def test_restore_tf(self):
        pass
    
if __name__ == "__main__":
    unittest.main()
