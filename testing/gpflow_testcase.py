import unittest
import tensorflow as tf

class GPflowTestCase(tf.test.TestCase):
    """
    Wrapper for TestCase to avoid massive duplication of resetting
    Tensorflow Graph.
    """

    _multiprocess_can_split_ = True

    def tearDown(self):
        tf.reset_default_graph()
        super(GPflowTestCase, self).tearDown()
