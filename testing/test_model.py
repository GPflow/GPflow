from __future__ import print_function
import GPflow
import tensorflow as tf
import numpy as np
import unittest


class TestOptimize(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        class Quadratic(GPflow.model.Model):
            def __init__(self):
                GPflow.model.Model.__init__(self)
                self.x = GPflow.param.Param(rng.randn(100))
            def build_likelihood(self):
                return -tf.reduce_sum(tf.square(self.x))
        self.m = Quadratic()

    def test_adam(self):
        o = tf.train.AdamOptimizer()
        self.m.optimize(o, max_iters=10000)
        self.failUnless(self.m.x._array.max() < 1e-3)
    
    def test_lbfgsb(self):
        self.m.optimize(display=False)
        self.failUnless(self.m.x._array.max() < 1e-6)


if __name__ == "__main__":
    unittest.main()

