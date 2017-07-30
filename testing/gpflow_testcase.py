import unittest
import tensorflow as tf

class GPflowTestCase(unittest.TestCase):
    """
    Wrapper for TestCase to avoid massive duplication of resetting
    Tensorflow Graph.
    """
    def tearDown(self):
        tf.reset_default_graph()
        super(GPflowTestCase, self).tearDown()
