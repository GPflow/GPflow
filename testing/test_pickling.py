from __future__ import print_function
import GPflow
import tensorflow as tf
import numpy as np
import unittest
import dill


class TestPickle(unittest.TestCase):
    """
    create, pickle and unpickle some simple objects (using dill)
    """
    def setUp(self):
        pass

    def test_param(self):
        # create a param object and pickle it
        p = GPflow.param.Param(np.arange(6).reshape(2, 3))
        with open('test_param.dill', 'wb') as f:
            dill.dump(p, f)
        with open('test_param.dill', 'rb') as f:
            pp = dill.load(f)

    def test_model(self):
        # create a model with a param and pickle
        m = GPflow.model.Model()
        m.p = GPflow.param.Param(np.arange(6).reshape(2, 3))

        with open('test_param.dill', 'wb') as f:
            dill.dump(m, f)
        with open('test_param.dill', 'rb') as f:
            mm = dill.load(f)


if __name__ == "__main__":
    unittest.main()
