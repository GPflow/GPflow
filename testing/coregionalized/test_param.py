# -*- coding: utf-8 -*-

import numpy as np
import GPflow
from GPflow.coregionalized.param import LabeledAutoFlow
from GPflow.coregionalized.labeled_data import LabeledData
import tensorflow as tf
import unittest
        
class DumbModel(GPflow.model.Model):
    def __init__(self):
        GPflow.model.Model.__init__(self)
        # LabeledAutoFlow cites self.num_labels        
        self.num_labels = 3
        self.a = GPflow.param.Param(3.)

    def build_likelihood(self):
        return -tf.square(self.a)

class NoArgsModel(DumbModel):
    @LabeledAutoFlow()
    def function(self):
        return self.a

class AddModel(DumbModel):
    @LabeledAutoFlow(LabeledData, LabeledData)
    def add(self, x, y):
        print(type(x.data), type(x.label))
        return tf.add(x.data, y.data)
        
        
class TestShareArgs(unittest.TestCase):
    """
    This is designed to replicate bug #85, where having two models caused
    autoflow functions to fail because the tf_args were shared over the
    instances.
    """
    def setUp(self):
        tf.reset_default_graph()
        self.m1 = AddModel()
        self.m1._compile()
        self.m2 = AddModel()
        self.m2._compile()
        rng = np.random.RandomState(0)
        self.x = rng.randn(10, 20)
        self.y = rng.randn(10, 20)
        self.label = rng.randint(0,3, 10)

    def test_share_args(self):
        self.m1.add((self.x, self.label), (self.y, self.label))
        self.m2.add((self.x, self.label), (self.y, self.label))
        self.m1.add((self.x, self.label), (self.y, self.label))


class TestAdd(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = AddModel()
        self.m._compile()
        rng = np.random.RandomState(0)
        self.x = rng.randn(10, 20)
        self.y = rng.randn(10, 20)
        self.label = rng.randint(0,3, 10)

    def test_add(self):
        self.assertTrue(np.allclose(self.x + self.y, \
            self.m.add((self.x, self.label), (self.y, self.label))))


if __name__ == "__main__":
    unittest.main()
