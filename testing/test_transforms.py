import GPflow
import tensorflow as tf
import numpy as np
import unittest


class TransformTests(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float64)
        self.x_np = np.random.randn(10)
        self.session = tf.Session()
        self.transforms = [C() for C in GPflow.transforms.Transform.__subclasses__()]
        self.transforms.append(GPflow.transforms.Logistic(7.3, 19.4))

    def test_tf_np_forward(self):
        """
        Make sure the np forward transforms are the same as the tensorflow ones
        """
        ys = [t.tf_forward(self.x) for t in self.transforms]
        ys_tf = [self.session.run(y, feed_dict={self.x: self.x_np}) for y in ys]
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        for y1, y2 in zip(ys_tf, ys_np):
            self.failUnless(np.allclose(y1, y2))

    def test_forward_backward(self):
        ys_np = [t.forward(self.x_np) for t in self.transforms]
        xs_np = [t.backward(y) for t, y in zip(self.transforms, ys_np)]
        for x in xs_np:
            self.failUnless(np.allclose(x, self.x_np))

    def test_logjac(self):
        """
        We have hand-crafted the log-jacobians for speed. Check they're correct
        wrt a tensorflow derived version
        """
        # there is no jacobian: loop manually
        def jacobian(f):
            return tf.pack([tf.gradients(f(self.x)[i], self.x)[0] for i in range(10)])
        tf_jacs = [tf.log(tf.matrix_determinant(jacobian(t.tf_forward))) for t in self.transforms]
        hand_jacs = [t.tf_log_jacobian(self.x) for t in self.transforms]

        for j1, j2 in zip(tf_jacs, hand_jacs):
            self.failUnless(np.allclose(self.session.run(j1, feed_dict={self.x: self.x_np}),
                                        self.session.run(j2, feed_dict={self.x: self.x_np})))


if __name__ == "__main__":
    unittest.main()
