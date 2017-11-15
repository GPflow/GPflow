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

import warnings
import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose


import gpflow
from gpflow.test_util import GPflowTestCase
from gpflow import settings


class TransformTests(GPflowTestCase):
    def prepare(self):
        x = tf.placeholder(settings.np_float, 10)
        x_np = np.random.randn(10).astype(settings.np_float)
        transforms = []
        for transform in gpflow.transforms.Transform.__subclasses__():
            if transform != gpflow.transforms.LowerTriangular:
                transforms.append(transform())
            else:
                transforms.append(transform(4))
        #self.transforms = [C() for C in gpflow.transforms.Transform.__subclasses__()]
        transforms.append(gpflow.transforms.Logistic(7.3, 19.4))
        return x, x_np, transforms

    def test_tf_np_forward(self):
        """
        Make sure the np forward transforms are the same as the tensorflow ones
        """
        with self.test_context() as sess:
            x, x_np, transforms = self.prepare()
            ys = [t.forward_tensor(x) for t in transforms]
            ys_tf = [sess.run(y, feed_dict={x: x_np}) for y in ys]
            ys_np = [t.forward(x_np) for t in transforms]
            for y1, y2 in zip(ys_tf, ys_np):
                assert_allclose(y1, y2)

    def test_forward_backward(self):
        with self.test_context():
            x, x_np, transforms = self.prepare()
            ys_np = [t.forward(x_np) for t in transforms]
            xs_np = [t.backward(y) for t, y in zip(transforms, ys_np)]
            for _t, x, _y in zip(transforms, xs_np, ys_np):
                assert_allclose(x.reshape(x_np.shape), x_np)
                # TODO(@awav): CHECK IT
                # self.assertTrue(t.free_state_size(y.shape) == len(x))

    def test_logjac(self):
        """
        We have hand-crafted the log-jacobians for speed. Check they're correct
        wrt a tensorflow derived version
        """
        with self.test_context() as sess:
            x, x_np, transforms = self.prepare()

            # there is no jacobian: loop manually
            def jacobian(f):
                return tf.stack([tf.gradients(f(x)[i], x)[0] for i in range(10)])

            tf_jacs = [tf.log(tf.matrix_determinant(jacobian(t.forward_tensor)))
                       for t in transforms
                       if not isinstance(t, gpflow.transforms.LowerTriangular)]
            hand_jacs = [t.log_jacobian_tensor(x)
                         for t in transforms
                         if not isinstance(t, gpflow.transforms.LowerTriangular)]

            for j1, j2 in zip(tf_jacs, hand_jacs):
                j1_res = sess.run(j1, feed_dict={x: x_np})
                j2_res = sess.run(j2, feed_dict={x: x_np})
                assert_allclose(j1_res, j2_res)


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
            self.assertEqual(len(w), 0)

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
    def testErrors(self):
        t = gpflow.transforms.LowerTriangular(1, 3)
        t.forward(np.ones(3 * 6))
        with self.assertRaises(ValueError):
            t.forward(np.ones(3 * 7))


class TestDiagMatrixTransform(GPflowTestCase):
    def setUp(self):
        self.t1 = gpflow.transforms.DiagMatrix(dim=1)
        self.t2 = gpflow.transforms.DiagMatrix(dim=3)

    def test_forward_backward(self):
        free_1d = np.random.randn(8)
        fwd1d = self.t1.forward(free_1d)
        assert_allclose(fwd1d.shape, np.array([len(free_1d), self.t1.dim, self.t1.dim]))
        assert_allclose(free_1d, self.t1.backward(fwd1d))

        size2d = 1
        free_2d = np.random.randn(size2d, self.t2.dim).flatten()
        fwd2d = self.t2.forward(free_2d)
        assert_allclose(fwd2d.shape, np.array([size2d, self.t2.dim, self.t2.dim]))
        assert_allclose(free_2d, self.t2.backward(fwd2d))

    def test_tf_np_forward(self):
        """
        Make sure the np forward transforms are the same as the tensorflow ones
        """
        with self.test_context() as sess:
            free = np.random.randn(8, self.t2.dim).flatten()
            x = tf.placeholder(settings.np_float)
            ys = sess.run(self.t2.forward_tensor(x), feed_dict={x: free})
            assert_allclose(ys, self.t2.forward(free))

            free = np.random.randn(1, self.t1.dim).flatten()
            x = tf.placeholder(settings.np_float)
            ys = sess.run(self.t1.forward_tensor(x), feed_dict={x: free})
            assert_allclose(ys, self.t1.forward(free))


if __name__ == "__main__":
    tf.test.main()
