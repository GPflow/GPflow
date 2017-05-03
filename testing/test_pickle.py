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

import unittest
import tensorflow as tf
import GPflow
import numpy as np
import pickle


class TestPickleEmpty(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = GPflow.model.Model()

    def test(self):
        s = pickle.dumps(self.m)
        pickle.loads(s)


class TestPickleSimple(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = GPflow.model.Model()
        self.m.p1 = GPflow.param.Param(np.random.randn(3, 2))
        self.m.p2 = GPflow.param.Param(np.random.randn(10))

    def test(self):
        s = pickle.dumps(self.m)
        m2 = pickle.loads(s)
        self.assertTrue(m2.p1._parent is m2)
        self.assertTrue(m2.p2._parent is m2)


class TestActiveDims(unittest.TestCase):
    def test(self):
        k = GPflow.kernels.RBF(2, active_dims=[0, 1])
        X = np.random.randn(10, 2)
        K = k.compute_K_symm(X)
        k = pickle.loads(pickle.dumps(k))
        K2 = k.compute_K_symm(X)
        self.assertTrue(np.allclose(K, K2))


class TestPickleGPR(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        self.m = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.RBF(1))

    def test(self):
        s1 = pickle.dumps(self.m)  # the model without running _compile
        self.m._compile()
        s2 = pickle.dumps(self.m)  # the model after _compile

        # reload the model
        m1 = pickle.loads(s1)
        m2 = pickle.loads(s2)
        m3 = pickle.loads(pickle.dumps(m1))

        # make sure the log likelihoods still match
        l1 = self.m.compute_log_likelihood()
        l2 = m1.compute_log_likelihood()
        l3 = m2.compute_log_likelihood()
        l4 = m3.compute_log_likelihood()
        self.assertTrue(l1 == l2 == l3 == l4)

        # make sure predictions still match (this tests AutoFlow)
        pX = np.linspace(-3, 3, 10)[:, None]
        p1, _ = self.m.predict_y(pX)
        p2, _ = m1.predict_y(pX)
        p3, _ = m2.predict_y(pX)
        p4, _ = m3.predict_y(pX)
        self.assertTrue(np.all(p1 == p2))
        self.assertTrue(np.all(p1 == p3))
        self.assertTrue(np.all(p1 == p4))


class TestPickleFix(unittest.TestCase):
    """
    Make sure a kernel with a fixed parameter can be computed after pickling
    """
    def test(self):
        k = GPflow.kernels.PeriodicKernel(1)
        k.period.fixed = True
        k = pickle.loads(pickle.dumps(k))
        x = np.linspace(0,1,100).reshape([-1,1])
        k.compute_K(x, x)

class TestPickleSVGP(unittest.TestCase):
    """
    Like the TestPickleGPR test, but with svgp (since it has extra tf variables
    for minibatching)
    """

    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        Z = rng.randn(5, 1)
        self.m = GPflow.svgp.SVGP(X, Y, Z=Z,
                                  likelihood=GPflow.likelihoods.Gaussian(),
                                  kern=GPflow.kernels.RBF(1))

    def test(self):
        s1 = pickle.dumps(self.m)  # the model without running _compile
        self.m._compile()
        s2 = pickle.dumps(self.m)  # the model after _compile

        # reload the model
        m1 = pickle.loads(s1)
        m2 = pickle.loads(s2)
        m3 = pickle.loads(pickle.dumps(m2))

        # make sure the log likelihoods still match
        l1 = self.m.compute_log_likelihood()
        l2 = m1.compute_log_likelihood()
        l3 = m2.compute_log_likelihood()
        l4 = m3.compute_log_likelihood()
        self.assertTrue(l1 == l2 == l3 == l4)

        # make sure predictions still match (this tests AutoFlow)
        pX = np.linspace(-3, 3, 10)[:, None]
        p1, _ = self.m.predict_y(pX)
        p2, _ = m1.predict_y(pX)
        p3, _ = m2.predict_y(pX)
        p4, _ = m3.predict_y(pX)
        self.assertTrue(np.all(p1 == p2))
        self.assertTrue(np.all(p1 == p3))
        self.assertTrue(np.all(p1 == p4))


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.transforms = GPflow.transforms.Transform.__subclasses__()
        self.models = []
        for T in self.transforms:
            m = GPflow.model.Model()
            m.x = GPflow.param.Param(1.0)
            m.x.transform = T()
            self.models.append(m)

    def test_pickle(self):
        strings = [pickle.dumps(m) for m in self.models]
        [pickle.loads(s) for s in strings]


if __name__ == "__main__":
    unittest.main()
