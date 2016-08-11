import unittest
import numpy as np
import tensorflow as tf
from GPflow.tf_hacks import vec_to_tri, tri_to_vec
import tensorflow.python.kernel_tests.gradient_checker as gc


def np_vec_to_tri(vec):
    ml = None
    for svec in vec:
        n = int(np.floor((vec.shape[1] * 8 + 1) ** 0.5 / 2.0 - 0.5))
        m = np.zeros((n, n))
        m[np.tril_indices(n, 0)] = svec
        ml = m[:, :, None] if ml is None else np.dstack((ml, m))
    return np.rollaxis(ml, 2, 0)


def compare_op(v):
    with tf.Session(''):
        return np_vec_to_tri(v) == vec_to_tri(v).eval()


class TestVecToTri(unittest.TestCase):
    def setUp(self):
        self.sess = tf.InteractiveSession()

    def testVecToTri(self):
        mats = [
            np.arange(1, 4)[None, :],
            np.vstack((np.arange(1, 4), np.arange(3, 0, -1))),
            np.arange(1, 16)[None, :]
        ]
        for m in mats:
            self.assertTrue(np.all(compare_op(m)))

    def testErrorOnIncorrectSize(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.sess.run(vec_to_tri(np.arange(1, 5)[None, :]))

    def testGradient(self):
        N = 30
        initval = np.arange(1, 0.5 * N * (N + 1) + 1)
        v = tf.Variable(initval[None, :])
        with tf.Session(''):
            f = (vec_to_tri(v) * np.random.randn(N, N)) ** 2.0  # Some function involving vec_to_tri
            self.assertLess(gc.compute_gradient_error(v, [1, len(initval)], f, [1, N, N]), 10 ** -10)


class TestTriToVec(unittest.TestCase):
    def setUp(self):
        self.sess = tf.InteractiveSession()

    def testTriToVec(self):
        mats = [
            np.arange(1, 4)[None, :],
            np.vstack((np.arange(1, 4), np.arange(3, 0, -1))),  # Currently, only do matrices
            np.arange(1, 16)[None, :]
        ]
        with tf.Session(''):
            for m in mats:
                # The ops are each others' inverse.
                self.assertTrue(np.all(tri_to_vec(vec_to_tri(m)).eval() == m))

    def test_wrong_shape(self):
        mats = [
            np.ones((2, 3, 4)),
            np.ones((3, 3))
        ]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            for m in mats:
                self.sess.run(tri_to_vec(m))


if __name__ == "__main__":
    unittest.main()
