# Copyright 2016 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.from __future__ import print_function

import gpflow
import tensorflow as tf
import numpy as np
import unittest
from gpflow import settings
import warnings

from testing.gpflow_testcase import GPflowTestCase
from gpflow import settings


float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class TransformTests(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.x = tf.placeholder(float_type, 10)
            self.x_np = np.random.randn(10).astype(np_float_type)
            self.transforms = []
            for transform in gpflow.transforms.Transform.__subclasses__():
                if transform!=gpflow.transforms.LowerTriangular:
                    self.transforms.append(transform())
                else:
                    self.transforms.append(transform(4))
            #self.transforms = [C() for C in gpflow.transforms.Transform.__subclasses__()]
            self.transforms.append(gpflow.transforms.Logistic(7.3, 19.4))

    def test_tf_np_forward(self):
        """
        Make sure the np forward transforms are the same as the tensorflow ones
        """
        with self.test_session() as sess:
            ys = [t.tf_forward(self.x) for t in self.transforms]
            ys_tf = [sess.run(y, feed_dict={self.x: self.x_np}) for y in ys]
            ys_np = [t.forward(self.x_np) for t in self.transforms]
            for y1, y2 in zip(ys_tf, ys_np):
                self.assertTrue(np.allclose(y1, y2))

    def test_forward_backward(self):
        with self.test_session() as sess:
            ys_np = [t.forward(self.x_np) for t in self.transforms]
            xs_np = [t.backward(y) for t, y in zip(self.transforms, ys_np)]
            for t, x, y in zip(self.transforms, xs_np, ys_np):
                self.assertTrue(np.allclose(x, self.x_np))
                self.assertTrue(t.free_state_size(y.shape) == len(x))

    def test_logjac(self):
        """
        We have hand-crafted the log-jacobians for speed. Check they're correct
        wrt a tensorflow derived version
        """

        # there is no jacobian: loop manually
        def jacobian(f):
            return tf.stack([tf.gradients(f(self.x)[i], self.x)[0] for i in range(10)])

        with self.test_session() as sess:
            tf_jacs = [tf.log(tf.matrix_determinant(jacobian(t.tf_forward)))
                       for t in self.transforms
                       if type(t) is not gpflow.transforms.LowerTriangular]
            hand_jacs = [t.tf_log_jacobian(self.x)
                         for t in self.transforms
                         if type(t) is not gpflow.transforms.LowerTriangular]

            for j1, j2 in zip(tf_jacs, hand_jacs):
                self.assertTrue(np.allclose(
                    sess.run(j1, feed_dict={self.x: self.x_np}),
                    sess.run(j2, feed_dict={self.x: self.x_np})))


class TestOverflow(GPflowTestCase):
    """
    Bug #302 identified an overflow in the standard positive transform. This is a regression test.
    """

    def setUp(self):
        self.t = gpflow.transforms.Log1pe()

    def testOverflow(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            y = self.t.forward(np.array([-1000, -300, -10, 10, 300, 1000]))
            self.assertTrue(len(w) == 0)

        self.assertFalse(np.any(np.isinf(y)))
        self.assertFalse(np.any(np.isnan(y)))

    def testDivByZero(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            y = self.t.backward(np.array([self.t._lower]))
            self.assertTrue(len(w) == 0)

        self.assertFalse(np.any(np.isinf(y)))
        self.assertFalse(np.any(np.isnan(y)))


class TestLowerTriTransform(GPflowTestCase):
    """
    Some extra tests for the LowerTriangle transformation.
    """

    def setUp(self):
        self.t = gpflow.transforms.LowerTriangular(1,3)

    def testErrors(self):
        self.t.free_state_size((6, 6, 3))
        with self.assertRaises(ValueError):
            self.t.free_state_size((6, 6, 2))
        with self.assertRaises(ValueError):
            self.t.free_state_size((7, 6, 3))

        self.t.forward(np.ones(3 * 6))
        with self.assertRaises(ValueError):
            self.t.forward(np.ones(3 * 7))


class TestDiagMatrixTransform(GPflowTestCase):
    def setUp(self):
        self.t1 = gpflow.transforms.DiagMatrix(dim=1)
        self.t2 = gpflow.transforms.DiagMatrix(dim=3)

    def test_forward_backward(self):
        free_1d = np.random.randn(8)
        fwd1d = self.t1.forward(free_1d)
        self.assertTrue(np.all(fwd1d.shape == np.array([len(free_1d), self.t1.dim, self.t1.dim])))
        self.assertTrue(np.allclose(free_1d, self.t1.backward(fwd1d)))

        size2d = 1
        free_2d = np.random.randn(size2d, self.t2.dim).flatten()
        fwd2d = self.t2.forward(free_2d)
        self.assertTrue(np.all(fwd2d.shape == np.array([size2d, self.t2.dim, self.t2.dim])))
        self.assertTrue(np.allclose(free_2d, self.t2.backward(fwd2d)))

    def test_tf_np_forward(self):
        """
        Make sure the np forward transforms are the same as the tensorflow ones
        """
        with self.test_session() as sess:
            free = np.random.randn(8, self.t2.dim).flatten()
            x = tf.placeholder(float_type)
            ys = sess.run(self.t2.tf_forward(x), feed_dict={x: free})
            self.assertTrue(np.allclose(ys, self.t2.forward(free)))

            free = np.random.randn(1, self.t1.dim).flatten()
            x = tf.placeholder(float_type)
            ys = sess.run(self.t1.tf_forward(x), feed_dict={x: free})
            self.assertTrue(np.allclose(ys, self.t1.forward(free)))


if __name__ == "__main__":
    unittest.main()
