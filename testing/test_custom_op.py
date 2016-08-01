import unittest
import numpy as np
import tensorflow as tf
from GPflow.tf_hacks import vec_to_tri


def np_vec_to_tri(vec):
    ml = None
    for svec in vec:
        n = int(np.floor((vec.shape[1] * 8 + 1)**0.5 / 2.0 - 0.5))
        m = np.zeros((n, n))
        m[np.tril_indices(n, 0)] = svec
        ml = m[:, :, None] if ml is None else np.dstack((ml, m))
    return np.rollaxis(ml, 2, 0)


def compare_op(v):
    with tf.Session(''):
        return np_vec_to_tri(v) == vec_to_tri(v).eval()


class TestVecToTri(unittest.TestCase):
    def testVecToTri(self):
        mats = [
            np.arange(1, 4)[None, :],
            np.vstack((np.arange(1, 4), np.arange(3, 0, -1))),
            np.arange(1, 16)[None, :]
        ]
        for m in mats:
            self.assertTrue(np.all(compare_op(m)))

    def testErrorOnIncorrectSize(self):
        def func():
            with tf.Session(''):
                vec_to_tri(np.arange(1, 5)[None, :]).eval()
        self.assertRaises(tf.errors.InvalidArgumentError, func)
