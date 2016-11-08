import GPflow
import tensorflow as tf
import numpy as np
import unittest
from GPflow import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class TestMeanFuncs(unittest.TestCase):
    """
    Test the output shape for basic and compositional mean functions, also
    check that the combination of mean functions returns the correct clas
    """
    def setUp(self):
        tf.reset_default_graph()
        self.input_dim = 3
        self.output_dim = 2
        self.N = 20
        rng = np.random.RandomState(0)
        self.mfs = [GPflow.mean_functions.Zero(),
                    GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim).astype(np_float_type), rng.randn(self.output_dim).astype(np_float_type)),
                    GPflow.mean_functions.Constant(rng.randn(self.output_dim).astype(np_float_type))]

        self.composition_mfs_add = []
        self.composition_mfs_mult = []

        for mean_f1 in self.mfs:
            self.composition_mfs_add.extend([mean_f1 + mean_f2 for mean_f2 in self.mfs])
            self.composition_mfs_mult.extend([mean_f1 * mean_f2 for mean_f2 in self.mfs])

        self.composition_mfs = self.composition_mfs_add + self.composition_mfs_mult
        self.x = tf.placeholder(float_type)

        for mf in self.mfs:
            mf.make_tf_array(self.x)

        self.X = tf.placeholder(float_type, [self.N, self.input_dim])
        self.X_data = np.random.randn(self.N, self.input_dim).astype(np_float_type)

    def test_basic_output_shape(self):
        for mf in self.mfs:
            with mf.tf_mode():
                Y = tf.Session().run(mf(self.X), feed_dict={self.x: mf.get_free_state(), self.X: self.X_data})
            self.assertTrue(Y.shape in [(self.N, self.output_dim), (self.N, 1)])

    def test_composition_output_shape(self):
        for comp_mf in self.composition_mfs:
            with comp_mf.tf_mode():
                Y = tf.Session().run(comp_mf(self.X), feed_dict={self.x: comp_mf.get_free_state(), self.X: self.X_data})
            self.assertTrue(Y.shape in [(self.N, self.output_dim), (self.N, 1)])

    def test_combination_types(self):
        self.assertTrue(all(isinstance(mfAdd, GPflow.mean_functions.Additive) for mfAdd in self.composition_mfs_add))
        self.assertTrue(all(isinstance(mfMult, GPflow.mean_functions.Product) for mfMult in self.composition_mfs_mult))


class TestModelCompositionOperations(unittest.TestCase):
    """
    Tests that operator precedence is correct and zero unary operations, i.e.
    adding 0, multiplying by 1, adding x and then subtracting etc. do not
    change the mean function
    """
    def setUp(self):
        tf.reset_default_graph()
        self.input_dim = 3
        self.output_dim = 2
        self.N = 20
        rng = np.random.RandomState(0)

        X = rng.randn(self.N, self.input_dim).astype(np_float_type)
        Y = rng.randn(self.N, self.output_dim).astype(np_float_type)
        self.Xtest = rng.randn(30, 3).astype(np_float_type)

        zero = GPflow.mean_functions.Zero()

        linear1 = GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim).astype(np_float_type), rng.randn(self.output_dim).astype(np_float_type))
        linear2 = GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim).astype(np_float_type), rng.randn(self.output_dim).astype(np_float_type))
        linear3 = GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim).astype(np_float_type), rng.randn(self.output_dim).astype(np_float_type))

        const1 = GPflow.mean_functions.Constant(rng.randn(self.output_dim).astype(np_float_type))
        const2 = GPflow.mean_functions.Constant(rng.randn(self.output_dim).astype(np_float_type))
        const3 = GPflow.mean_functions.Constant(rng.randn(self.output_dim).astype(np_float_type))

        const1inv = GPflow.mean_functions.Constant(np.reshape(const1.c.get_free_state() * -1, [self.output_dim]))
        linear1inv = GPflow.mean_functions.Linear(A=np.reshape(linear1.A.get_free_state() * -1., [self.input_dim, self.output_dim]),
                                                  b=np.reshape(linear1.b.get_free_state() * -1., [self.output_dim]))

        # a * (b + c)
        const_set1 = GPflow.mean_functions.Product(const1,
                                                   GPflow.mean_functions.Additive(const2, const3))
        linear_set1 = GPflow.mean_functions.Product(linear1,
                                                    GPflow.mean_functions.Additive(linear2, linear3))

        # ab + ac
        const_set2 = GPflow.mean_functions.Additive(GPflow.mean_functions.Product(const1, const2),
                                                    GPflow.mean_functions.Product(const1, const3))

        linear_set2 = GPflow.mean_functions.Additive(GPflow.mean_functions.Product(linear1, linear2),
                                                     GPflow.mean_functions.Product(linear1, linear3))
        # a-a = 0, (a + b) -a = b = a + (b - a)

        linear1_minus_linear1 = GPflow.mean_functions.Additive(linear1, linear1inv)
        const1_minus_const1 = GPflow.mean_functions.Additive(const1, const1inv)

        comp_minus_constituent1 = GPflow.mean_functions.Additive(GPflow.mean_functions.Additive(linear1, linear2),
                                                                      linear1inv)
        comp_minus_constituent2 = GPflow.mean_functions.Additive(linear1,
                                                                      GPflow.mean_functions.Additive(linear2,
                                                                                                     linear1inv))

        k = GPflow.kernels.Bias(self.input_dim)

        self.m_linear_set1 = GPflow.gpr.GPR(X, Y, mean_function=linear_set1, kern=k)
        self.m_linear_set2 = GPflow.gpr.GPR(X, Y, mean_function=linear_set2, kern=k)

        self.m_const_set1 = GPflow.gpr.GPR(X, Y, mean_function=const_set1, kern=k)
        self.m_const_set2 = GPflow.gpr.GPR(X, Y, mean_function=const_set2, kern=k)

        self.m_linear_min_linear = GPflow.gpr.GPR(X, Y, mean_function=linear1_minus_linear1, kern=k)
        self.m_const_min_const = GPflow.gpr.GPR(X, Y, mean_function=const1_minus_const1, kern=k)

        self.m_constituent = GPflow.gpr.GPR(X, Y, mean_function=linear2, kern=k)
        self.m_zero = GPflow.gpr.GPR(X, Y, mean_function=zero, kern=k)

        self.m_comp_minus_constituent1 = GPflow.gpr.GPR(X, Y, mean_function=comp_minus_constituent1, kern=k)
        self.m_comp_minus_constituent2 = GPflow.gpr.GPR(X, Y, mean_function=comp_minus_constituent2, kern=k)

    def test_precedence(self):
        mu1_lin, v1_lin = self.m_linear_set1.predict_f(self.Xtest)
        mu2_lin, v2_lin = self.m_linear_set2.predict_f(self.Xtest)

        mu1_const, v1_const = self.m_const_set1.predict_f(self.Xtest)
        mu2_const, v2_const = self.m_const_set2.predict_f(self.Xtest)

        self.assertTrue(np.all(np.isclose(v1_lin, v1_lin)))
        self.assertTrue(np.all(np.isclose(mu1_lin, mu2_lin)))

        self.assertTrue(np.all(np.isclose(v1_const, v2_const)))
        self.assertTrue(np.all(np.isclose(mu1_const, mu2_const)))

    def test_inverse_operations(self):
        mu1_lin_min_lin, v1_lin_min_lin = self.m_linear_min_linear.predict_f(self.Xtest)
        mu1_const_min_const, v1_const_min_const = self.m_const_min_const.predict_f(self.Xtest)

        mu1_comp_min_constituent1, v1_comp_min_constituent1 = self.m_comp_minus_constituent1.predict_f(self.Xtest)
        mu1_comp_min_constituent2, v1_comp_min_constituent2 = self.m_comp_minus_constituent2.predict_f(self.Xtest)

        mu_const, _ = self.m_constituent.predict_f(self.Xtest)
        mu_zero, v_zero = self.m_zero.predict_f(self.Xtest)

        self.assertTrue(np.all(np.isclose(mu1_lin_min_lin, mu_zero)))
        self.assertTrue(np.all(np.isclose(mu1_const_min_const, mu_zero)))

        self.assertTrue(np.all(np.isclose(mu1_comp_min_constituent1, mu_const)))
        self.assertTrue(np.all(np.isclose(mu1_comp_min_constituent2, mu_const)))
        self.assertTrue(np.all(np.isclose(mu1_comp_min_constituent1, mu1_comp_min_constituent2)))


class TestModelsWithMeanFuncs(unittest.TestCase):
    """
    Simply check that all models have a higher prediction with a constant mean
    function than with a zero mean function.

    For compositions of mean functions check that multiplication/ addition of
    a constant results in a higher prediction, whereas addition of zero/
    mutliplication with one does not.

    """

    def setUp(self):
        tf.reset_default_graph()
        self.input_dim = 3
        self.output_dim = 2
        self.N = 20
        self.Ntest = 30
        self.M = 5
        rng = np.random.RandomState(0)
        X, Y, Z, self.Xtest = rng.randn(self.N, self.input_dim).astype(np_float_type),\
                              rng.randn(self.N, self.output_dim).astype(np_float_type),\
                              rng.randn(self.M, self.input_dim).astype(np_float_type),\
                              rng.randn(self.Ntest, self.input_dim).astype(np_float_type)
        k = lambda: GPflow.kernels.Matern32(self.input_dim)
        lik = lambda: GPflow.likelihoods.Gaussian()

        # test all models with these mean functions
        mf0 = GPflow.mean_functions.Zero()
        mf1 = GPflow.mean_functions.Constant(np.ones(self.output_dim) * 10)

        self.models_with, self.models_without = \
                [[GPflow.gpr.GPR(X, Y, mean_function=mf, kern=k()),
                  GPflow.sgpr.SGPR(X, Y, mean_function=mf, Z=Z, kern=k()),
                  GPflow.sgpr.GPRFITC(X, Y, mean_function=mf, Z=Z, kern=k()),
                  GPflow.svgp.SVGP(X, Y, mean_function=mf, Z=Z, kern=k(), likelihood=lik()),
                  GPflow.vgp.VGP(X, Y, mean_function=mf, kern=k(), likelihood=lik()),
                  GPflow.vgp.VGP(X, Y, mean_function=mf, kern=k(), likelihood=lik()),
                  GPflow.gpmc.GPMC(X, Y, mean_function=mf, kern=k(), likelihood=lik()),
                  GPflow.sgpmc.SGPMC(X, Y, mean_function=mf, kern=k(), likelihood=lik(), Z=Z)] for mf in (mf0, mf1)]

    def test_basic_mean_function(self):
        for m_with, m_without in zip(self.models_with, self.models_without):
            mu1, v1 = m_with.predict_f(self.Xtest)
            mu2, v2 = m_without.predict_f(self.Xtest)
            self.assertTrue(np.all(v1 == v2))
            self.assertFalse(np.all(mu1 == mu2))


class TestSwitchedMeanFunction(unittest.TestCase):
    """
    Test for the SwitchedMeanFunction.
    """
    def setUp(self):
        pass

    def test(self):
        rng = np.random.RandomState(0)
        X = np.hstack([rng.randn(10, 3), 1.0*rng.randint(0, 2, 10).reshape(-1, 1)])
        switched_mean = GPflow.mean_functions.SwitchedMeanFunction(
                        [GPflow.mean_functions.Constant(np.zeros(1)),
                         GPflow.mean_functions.Constant(np.ones(1))])

        sess = tf.Session()
        tf_array = switched_mean.get_free_state()
        switched_mean.make_tf_array(tf_array)
        sess.run(tf.initialize_all_variables())
        fd = {}
        switched_mean.update_feed_dict(switched_mean.get_feed_dict_keys(), fd)
        with switched_mean.tf_mode():
            result = sess.run(switched_mean(X), feed_dict=fd)

        np_list = np.array([0., 1.])
        result_ref = (np_list[X[:, 3].astype(np.int)]).reshape(-1, 1)
        self.assertTrue(np.allclose(result, result_ref))


if __name__ == "__main__":
    unittest.main()
