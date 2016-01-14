import GPflow
import theano
import numpy as np
import unittest
from theano import tensor as tt
import scipy

class JitcholTest(unittest.TestCase):
    def setUp(self):
        self.singular = np.ones((2,2))
        self.just_pos_def = self.singular + np.eye(2) * 1e-15
    def test_singular(self):
        with self.assertRaises(scipy.linalg.LinAlgError):
            GPflow.slinalg.jitchol(self.singular)
        #make sure this does not error:
        GPflow.slinalg.jitchol(self.just_pos_def)


       

    


if __name__ == "__main__":
    unittest.main()

