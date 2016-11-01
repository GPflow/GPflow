import tensorflow as tf
import numpy as np
import unittest
import GPflow.etransforms as transforms


class TestBlockTriDiagonalTransform(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float64)
        self.x_np = np.random.randn(7, 4, 2)
        self.session = tf.Session()
        self.transforms = [transforms.TriDiagonalBlockRep()]

    def test_tf_np_forward(self):
        """
        Make sure the np forward transforms are the same as the tensorflow ones
        """
        ys = [t.tf_forward(self.x) for t in self.transforms]
        ys_tf = [self.session.run(y, feed_dict={self.x: self.x_np}) for y in ys]
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        for y1, y2 in zip(ys_tf, ys_np):
            self.assertTrue(np.allclose(y1, y2))

    def test_fullmat(self):
        """
        Make sure the full matrix representation is PSD.
        """
        for _ in range(100):
            self.x_np = np.random.randn(7, 4, 2)
            fullmat = self.transforms[0].forward_fullmat(self.x_np)
            self.assertAlmostEqual(np.max(np.abs(fullmat - fullmat.T)), 0.0, msg="Matrix not symmetric.")
            self.assertTrue(np.linalg.det(self.transforms[0].forward_fullmat(self.x_np)) > 0, "Matrix not PD")

    def test_sample(self):
        ns = 1000000
        t = transforms.TriDiagonalBlockRep()
        bm = t.forward(self.x_np)
        s = t.sample(bm, ns)
        sr = s.reshape(ns, -1)
        fm = t.forward_fullmat(self.x_np)
        empcov1 = np.einsum('ni,nj->ij', sr, sr) / ns
        self.assertTrue(np.allclose(empcov1, fm, 0.05, 2e-2))


if __name__ == "__main__":
    unittest.main()
