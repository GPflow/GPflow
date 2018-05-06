# Copyright 2017 the GPflow authors.
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
# limitations under the License.

import itertools
import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose


import gpflow
from gpflow.test_util import GPflowTestCase
from gpflow import settings


class TestMeanFuncs(GPflowTestCase):
    """
    Test the output shape for basic and compositional mean functions, also
    check that the combination of mean functions returns the correct clas
    """

    input_dim = 3
    output_dim = 2
    N = 20

    def mfs1(self):
        rng = np.random.RandomState(0)
        return [gpflow.mean_functions.Zero(),
                gpflow.mean_functions.Linear(
                    rng.randn(self.input_dim, self.output_dim).astype(settings.float_type),
                    rng.randn(self.output_dim).astype(settings.float_type)),
                gpflow.mean_functions.Constant(
                    rng.randn(self.output_dim).astype(settings.float_type))]

    def mfs2(self):
        rng = np.random.RandomState(0)
        return [gpflow.mean_functions.Zero(),
                gpflow.mean_functions.Linear(
                    rng.randn(self.input_dim, self.output_dim).astype(settings.float_type),
                    rng.randn(self.output_dim).astype(settings.float_type)),
                gpflow.mean_functions.Constant(
                    rng.randn(self.output_dim).astype(settings.float_type))]

    def composition_mfs_add(self):
        composition_mfs_add = []
        for (mean_f1, mean_f2) in itertools.product(self.mfs1(), self.mfs2()):
            composition_mfs_add.extend([mean_f1 + mean_f2])
        return composition_mfs_add

    def composition_mfs_mult(self):
        composition_mfs_mult = []
        for (mean_f1, mean_f2) in itertools.product(self.mfs1(), self.mfs2()):
            composition_mfs_mult.extend([mean_f1 * mean_f2])
        return composition_mfs_mult

    def composition_mfs(self):
        return self.composition_mfs_add() + self.composition_mfs_mult()

    def test_basic_output_shape(self):
        with self.test_context() as sess:
            X = tf.placeholder(settings.float_type, shape=[self.N, self.input_dim])
            X_data = np.random.randn(self.N, self.input_dim).astype(settings.float_type)
            for mf in self.mfs1():
                mf.compile()
                Y = sess.run(mf(X), feed_dict={X: X_data})
                self.assertTrue(Y.shape in [(self.N, self.output_dim), (self.N, 1)])

    def test_add_output_shape(self):
        with self.test_context() as sess:
            X = tf.placeholder(settings.float_type, [self.N, self.input_dim])
            X_data = np.random.randn(self.N, self.input_dim).astype(settings.float_type)
            for comp_mf in self.composition_mfs_add():
                comp_mf.compile()
                Y = sess.run(comp_mf(X), feed_dict={X: X_data})
                self.assertTrue(Y.shape in [(self.N, self.output_dim), (self.N, 1)])

    def test_mult_output_shape(self):
        with self.test_context() as sess:
            X = tf.placeholder(settings.float_type, [self.N, self.input_dim])
            X_data = np.random.randn(self.N, self.input_dim).astype(settings.float_type)
            for comp_mf in self.composition_mfs_mult():
                comp_mf.compile()
                Y = sess.run(comp_mf(X), feed_dict={X: X_data})
                self.assertTrue(Y.shape in [(self.N, self.output_dim), (self.N, 1)])

    def test_composition_output_shape(self):
        with self.test_context() as sess:
            X = tf.placeholder(settings.float_type, [self.N, self.input_dim])
            X_data = np.random.randn(self.N, self.input_dim).astype(settings.float_type)
            comp_mf = self.composition_mfs()[1]
            comp_mf.compile()
            # for comp_mf in self.composition_mfs:
            Y = sess.run(comp_mf(X), feed_dict={X: X_data})
            self.assertTrue(Y.shape in [(self.N, self.output_dim), (self.N, 1)])

    def test_combination_types(self):
        with self.test_context():
            self.assertTrue(all(isinstance(mf, gpflow.mean_functions.Additive)
                                for mf in self.composition_mfs_add()))
            self.assertTrue(all(isinstance(mf, gpflow.mean_functions.Product)
                                for mf in self.composition_mfs_mult()))


class TestModelCompositionOperations(GPflowTestCase):
    """
    Tests that operator precedence is correct and zero unary operations, i.e.
    adding 0, multiplying by 1, adding x and then subtracting etc. do not
    change the mean function
    """

    input_dim = 3
    output_dim = 2
    N = 20
    rng = np.random.RandomState(0)

    Xtest = rng.randn(30, 3).astype(settings.float_type)

    def initials(self):
        # Need two copies of the linear1_1 (l1, l2) since we can't add the
        # same parameter twice to a single tree.
        self.rng.seed(seed=0)
        linear1_1 = gpflow.mean_functions.Linear(
            self.rng.randn(self.input_dim, self.output_dim).astype(settings.float_type),
            self.rng.randn(self.output_dim).astype(settings.float_type))
        self.rng.seed(seed=0)
        linear1_2 = gpflow.mean_functions.Linear(
            self.rng.randn(self.input_dim, self.output_dim).astype(settings.float_type),
            self.rng.randn(self.output_dim).astype(settings.float_type))
        self.rng.seed(seed=1)
        linear2 = gpflow.mean_functions.Linear(
            self.rng.randn(self.input_dim, self.output_dim).astype(settings.float_type),
            self.rng.randn(self.output_dim).astype(settings.float_type))
        linear3 = gpflow.mean_functions.Linear(
            self.rng.randn(self.input_dim, self.output_dim).astype(settings.float_type),
            self.rng.randn(self.output_dim).astype(settings.float_type))

        linears = (linear1_1, linear1_2, linear2, linear3)

        # Need two copies of the const1 since we can't add the same parameter
        # twice to a single tree
        self.rng.seed(seed=2)
        const1_1 = gpflow.mean_functions.Constant(
            self.rng.randn(self.output_dim).astype(settings.float_type))
        self.rng.seed(seed=2)
        const1_2 = gpflow.mean_functions.Constant(
            self.rng.randn(self.output_dim).astype(settings.float_type))
        self.rng.seed(seed=3)
        const2 = gpflow.mean_functions.Constant(
            self.rng.randn(self.output_dim).astype(settings.float_type))
        const3 = gpflow.mean_functions.Constant(
            self.rng.randn(self.output_dim).astype(settings.float_type))

        consts = (const1_1, const1_2, const2, const3)

        const1inv = gpflow.mean_functions.Constant(const1_1.c.read_value() * -1)
        linear1inv = gpflow.mean_functions.Linear(
            A=(linear1_1.A.read_value() * -1.),
            b=(linear1_2.b.read_value() * -1.))

        invs = (linear1inv, const1inv)

        return linears, consts, invs

    def a_b_plus_c(self):
        # a * (b + c)

        linears, consts, _ = self.initials()
        linear1_1, _linear1_2, linear2, linear3 = linears
        const1_1, _const1_2, const2, const3 = consts

        const_set1 = gpflow.mean_functions.Product(
            const1_1, gpflow.mean_functions.Additive(const2, const3))

        linear_set1 = gpflow.mean_functions.Product(
            linear1_1, gpflow.mean_functions.Additive(linear2, linear3))

        return const_set1, linear_set1

    def ab_plus_ac(self):
        linears, consts, _ = self.initials()
        linear1_1, linear1_2, linear2, linear3 = linears
        const1_1, const1_2, const2, const3 = consts

        # ab + ac
        const_set2 = gpflow.mean_functions.Additive(
            gpflow.mean_functions.Product(const1_1, const2),
            gpflow.mean_functions.Product(const1_2, const3))

        linear_set2 = gpflow.mean_functions.Additive(
            gpflow.mean_functions.Product(linear1_1, linear2),
            gpflow.mean_functions.Product(linear1_2, linear3))

        return const_set2, linear_set2


        # a-a = 0,
    def a_minus_a(self):

        linears, consts, invs = self.initials()
        linear1_1, _linear1_2, _linear2, _linear3 = linears
        const1_1, _const1_2, _const2, _const3 = consts
        linear1inv, const1inv = invs

        linear1_minus_linear1 = gpflow.mean_functions.Additive(linear1_1, linear1inv)
        const1_minus_const1 = gpflow.mean_functions.Additive(const1_1, const1inv)
        return linear1_minus_linear1, const1_minus_const1

    def comp_minus_constituent1(self):
        # (a + b) - a = b = a + (b - a)

        linears, _consts, invs = self.initials()
        linear1_1, _linear1_2, linear2, _linear3 = linears
        linear1inv, _const1inv = invs

        comp_minus_constituent1 = gpflow.mean_functions.Additive(
            gpflow.mean_functions.Additive(linear1_1, linear2), linear1inv)

        return comp_minus_constituent1

    def comp_minus_constituent2(self):

        linears, _consts, invs = self.initials()
        linear1_1, _linear1_2, linear2, _linear3 = linears
        linear1inv, _const1inv = invs

        comp_minus_constituent2 = gpflow.mean_functions.Additive(
            linear1_1, gpflow.mean_functions.Additive(linear2, linear1inv))

        return comp_minus_constituent2

    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context():
            zero = gpflow.mean_functions.Zero()
            k = gpflow.kernels.Bias(self.input_dim)

            const_set1, linear_set1 = self.a_b_plus_c()
            const_set2, linear_set2 = self.ab_plus_ac()
            linear1_minus_linear1, const1_minus_const1 = self.a_minus_a()
            comp_minus_constituent1 = self.comp_minus_constituent1()
            comp_minus_constituent2 = self.comp_minus_constituent2()
            _linear1_1, _linear1_2, linear2, _linear3 = self.initials()[0]

            X = self.rng.randn(self.N, self.input_dim).astype(settings.float_type)
            Y = self.rng.randn(self.N, self.output_dim).astype(settings.float_type)

            self.m_linear_set1 = gpflow.models.GPR(X, Y, mean_function=linear_set1, kern=k)
            self.m_linear_set2 = gpflow.models.GPR(X, Y, mean_function=linear_set2, kern=k)

            self.m_const_set1 = gpflow.models.GPR(X, Y, mean_function=const_set1, kern=k)
            self.m_const_set2 = gpflow.models.GPR(X, Y, mean_function=const_set2, kern=k)

            self.m_linear_min_linear = gpflow.models.GPR(
                X, Y, mean_function=linear1_minus_linear1, kern=k)
            self.m_const_min_const = gpflow.models.GPR(
                X, Y, mean_function=const1_minus_const1, kern=k)

            self.m_constituent = gpflow.models.GPR(X, Y, mean_function=linear2, kern=k)
            self.m_zero = gpflow.models.GPR(X, Y, mean_function=zero, kern=k)

            self.m_comp_minus_constituent1 = gpflow.models.GPR(
                X, Y, mean_function=comp_minus_constituent1, kern=k)
            self.m_comp_minus_constituent2 = gpflow.models.GPR(
                X, Y, mean_function=comp_minus_constituent2, kern=k)

    def test_precedence(self):
        with self.test_context():
            mu1_lin, v1_lin = self.m_linear_set1.predict_f(self.Xtest)
            mu2_lin, v2_lin = self.m_linear_set2.predict_f(self.Xtest)

            mu1_const, v1_const = self.m_const_set1.predict_f(self.Xtest)
            mu2_const, v2_const = self.m_const_set2.predict_f(self.Xtest)

            assert_allclose(v1_lin, v1_lin)
            assert_allclose(mu1_lin, mu2_lin)

            assert_allclose(v1_const, v2_const)
            assert_allclose(mu1_const, mu2_const)

    def test_inverse_operations(self):
        with self.test_context():
            mu1_lin_min_lin, v1_lin_min_lin = self.m_linear_min_linear.predict_f(self.Xtest)

            mu1_const_min_const, v1_const_min_const = self.m_const_min_const.predict_f(self.Xtest)

            mu1_comp_min_constituent1, v1_comp_min_constituent1 = self.m_comp_minus_constituent1.predict_f(self.Xtest)

            mu1_comp_min_constituent2, v1_comp_min_constituent2 = self.m_comp_minus_constituent2.predict_f(self.Xtest)

            mu_const, _ = self.m_constituent.predict_f(self.Xtest)
            mu_zero, v_zero = self.m_zero.predict_f(self.Xtest)

            self.assertTrue(np.all(np.isclose(mu1_lin_min_lin, mu_zero)))
            self.assertTrue(np.all(np.isclose(mu1_const_min_const, mu_zero)))

            assert_allclose(mu1_comp_min_constituent1, mu_const)
            assert_allclose(mu1_comp_min_constituent2, mu_const)
            assert_allclose(mu1_comp_min_constituent1, mu1_comp_min_constituent2)


class TestModelsWithMeanFuncs(GPflowTestCase):
    """
    Simply check that all models have a higher prediction with a constant mean
    function than with a zero mean function.

    For compositions of mean functions check that multiplication/ addition of
    a constant results in a higher prediction, whereas addition of zero/
    mutliplication with one does not.

    """

    def setUp(self):
        self.input_dim = 3
        self.output_dim = 2
        self.N = 20
        self.Ntest = 30
        self.M = 5
        rng = np.random.RandomState(0)
        X, Y, Z, self.Xtest = (
            rng.randn(self.N, self.input_dim).astype(settings.float_type),
            rng.randn(self.N, self.output_dim).astype(settings.float_type),
            rng.randn(self.M, self.input_dim).astype(settings.float_type),
            rng.randn(self.Ntest, self.input_dim).astype(settings.float_type))

        with self.test_context():
            k = lambda: gpflow.kernels.Matern32(self.input_dim)
            lik = lambda: gpflow.likelihoods.Gaussian()
            mf0 = lambda: gpflow.mean_functions.Zero()
            mf1 = lambda: gpflow.mean_functions.Constant(np.ones(self.output_dim) * 10)

            self.models_with, self.models_without = ([
                [gpflow.models.GPR(X, Y, mean_function=mf(), kern=k()),
                 gpflow.models.SGPR(X, Y, mean_function=mf(), Z=Z, kern=k()),
                 gpflow.models.GPRFITC(X, Y, mean_function=mf(), Z=Z, kern=k()),
                 gpflow.models.SVGP(X, Y, mean_function=mf(), Z=Z, kern=k(), likelihood=lik()),
                 gpflow.models.VGP(X, Y, mean_function=mf(), kern=k(), likelihood=lik()),
                 gpflow.models.VGP(X, Y, mean_function=mf(), kern=k(), likelihood=lik()),
                 gpflow.models.GPMC(X, Y, mean_function=mf(), kern=k(), likelihood=lik()),
                 gpflow.models.SGPMC(X, Y, mean_function=mf(), kern=k(), likelihood=lik(), Z=Z)]
                for mf in (mf0, mf1)])

    def test_basic_mean_function(self):
        with self.test_context():
            for m_with, m_without in zip(self.models_with, self.models_without):
                m_with.compile()
                m_without.compile()
                mu1, v1 = m_with.predict_f(self.Xtest)
                mu2, v2 = m_without.predict_f(self.Xtest)
                self.assertTrue(np.all(v1 == v2))
                self.assertFalse(np.all(mu1 == mu2))


class TestSwitchedMeanFunction(GPflowTestCase):
    """
    Test for the SwitchedMeanFunction.
    """

    def test(self):
        with self.test_context() as sess:
            rng = np.random.RandomState(0)
            X = np.hstack([rng.randn(10, 3), 1.0 * rng.randint(0, 2, 10).reshape(-1, 1)])
            zeros = gpflow.mean_functions.Constant(np.zeros(1))
            ones = gpflow.mean_functions.Constant(np.ones(1))
            switched_mean = gpflow.mean_functions.SwitchedMeanFunction([zeros, ones])
            switched_mean.compile()
            result = sess.run(switched_mean(X))
            np_list = np.array([0., 1.])
            result_ref = (np_list[X[:, 3].astype(np.int)]).reshape(-1, 1)
            assert_allclose(result, result_ref)


class TestBug277Regression(GPflowTestCase):
    """
    See github issue #277. This is a regression test.
    """
    def test(self):
        with self.test_context():
            m1 = gpflow.mean_functions.Linear()
            m2 = gpflow.mean_functions.Linear()
            self.assertTrue(m1.b.read_value() == m2.b.read_value())
            m1.b = [1.]
            self.assertFalse(m1.b.read_value() == m2.b.read_value())


if __name__ == "__main__":
    tf.test.main()
