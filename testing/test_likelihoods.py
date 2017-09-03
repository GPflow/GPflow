import unittest
import six
import tensorflow as tf
import numpy as np

import gpflow
from gpflow import settings

from testing.gpflow_testcase import GPflowTestCase

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class TestSetup(object):
    def __init__(self, likelihood, Y, tolerance):
        self.likelihood, self.Y, self.tolerance = likelihood, Y, tolerance
        self.is_analytic = six.get_unbound_function(likelihood.predict_density) is not\
            six.get_unbound_function(gpflow.likelihoods.Likelihood.predict_density)


def getTestSetups(includeMultiClass=True, addNonStandardLinks=False):
    test_setups = []
    rng = np.random.RandomState(1)
    for likelihoodClass in gpflow.likelihoods.Likelihood.__subclasses__():
        if likelihoodClass == gpflow.likelihoods.Ordinal:
            test_setups.append(TestSetup(likelihoodClass(np.array([-1, 1])), rng.randint(0, 3, (10, 2)), 1e-6))
        elif likelihoodClass == gpflow.likelihoods.SwitchedLikelihood:
            continue  # switched likelihood tested separately
        elif (likelihoodClass == gpflow.likelihoods.MultiClass):
            if includeMultiClass:
                sample = rng.randn(10, 2)
                # Multiclass needs a less tight tolerance due to presence of clipping.
                tolerance = 1e-3
                test_setups.append(TestSetup(likelihoodClass(2),  np.argmax(sample, 1).reshape(-1, 1), tolerance))
        else:
            # most likelihoods follow this standard:
            test_setups.append(TestSetup(likelihoodClass(), rng.rand(10, 2).astype(np_float_type), 1e-6))

    if addNonStandardLinks:
        test_setups.append(TestSetup(gpflow.likelihoods.Poisson(invlink=tf.square),
                                     rng.rand(10, 2).astype(np_float_type), 1e-6))
        test_setups.append(TestSetup(gpflow.likelihoods.Exponential(invlink=tf.square),
                                     rng.rand(10, 2).astype(np_float_type), 1e-6))
        test_setups.append(TestSetup(gpflow.likelihoods.Gamma(invlink=tf.square),
                                     rng.rand(10, 2).astype(np_float_type), 1e-6))

        def sigmoid(x):
            return 1./(1 + tf.exp(-x))
        test_setups.append(TestSetup(gpflow.likelihoods.Bernoulli(invlink=sigmoid),
                                     rng.rand(10, 2).astype(np_float_type), 1e-6))
    return test_setups


class TestPredictConditional(GPflowTestCase):
    """
    Here we make sure that the conditional_mean and contitional_var functions
    give the same result as the predict_mean_and_var function if the prediction
    has no uncertainty.
    """
    def setUp(self):
        with self.test_session():
            self.test_setups = getTestSetups(addNonStandardLinks=True)
            self.x = tf.placeholder(float_type)
            for test_setup in self.test_setups:
                test_setup.likelihood.make_tf_array(self.x)
            self.F = tf.placeholder(float_type)
            rng = np.random.RandomState(0)
            self.F_data = rng.randn(10, 2).astype(np_float_type)

    def test_mean(self):
        with self.test_session() as sess:
            for test_setup in self.test_setups:
                l = test_setup.likelihood
                with l.tf_mode():
                    mu1 = sess.run(
                        l.conditional_mean(self.F),
                        feed_dict={self.x: l.get_free_state(), self.F: self.F_data})
                    mu2, _ = sess.run(
                        l.predict_mean_and_var(self.F, self.F * 0),
                        feed_dict={self.x: l.get_free_state(), self.F: self.F_data})
                self.assertTrue(np.allclose(mu1, mu2, test_setup.tolerance, test_setup.tolerance))

    def test_variance(self):
        with self.test_session() as sess:
            for test_setup in self.test_setups:
                l = test_setup.likelihood
                with l.tf_mode():
                    v1 = sess.run(
                        l.conditional_variance(self.F),
                        feed_dict={self.x: l.get_free_state(), self.F: self.F_data})
                    v2 = sess.run(
                        l.predict_mean_and_var(self.F, self.F * 0)[1],
                        feed_dict={self.x: l.get_free_state(), self.F: self.F_data})
                self.assertTrue(np.allclose(v1, v2, atol=test_setup.tolerance))

    def test_var_exp(self):
        """
        Here we make sure that the variational_expectations gives the same result
        as logp if the latent function has no uncertainty.
        """
        with self.test_session() as sess:
            for test_setup in self.test_setups:
                l = test_setup.likelihood
                y = test_setup.Y
                with l.tf_mode():
                    r1 = sess.run(
                        l.logp(self.F, y),
                        feed_dict={self.x: l.get_free_state(), self.F: self.F_data})
                    r2 = sess.run(
                        l.variational_expectations(self.F, self.F * 0, test_setup.Y),
                        feed_dict={self.x: l.get_free_state(), self.F: self.F_data})
                self.assertTrue(np.allclose(r1, r2, test_setup.tolerance, test_setup.tolerance))


class TestQuadrature(GPflowTestCase):
    """
    Where quadratre methods have been overwritten, make sure the new code
     does something close to the quadrature
    """
    def setUp(self):
        with self.test_session() as sess:
            self.rng = np.random.RandomState()
            self.Fmu, self.Fvar, self.Y = self.rng.randn(3, 10, 2).astype(np_float_type)
            self.Fvar = 0.01 * self.Fvar ** 2
            self.test_setups = getTestSetups(includeMultiClass=False)

    def test_var_exp(self):
        with self.test_session() as sess:
            # get all the likelihoods where variational expectations has been overwritten
            for test_setup in self.test_setups:
                if not test_setup.is_analytic:
                    continue
                l = test_setup.likelihood
                y = test_setup.Y
                x_data = l.get_free_state()
                x = tf.placeholder(float_type)
                l.make_tf_array(x)
                # 'build' the functions
                with l.tf_mode():
                    F1 = l.variational_expectations(self.Fmu, self.Fvar, y)
                    F2 = gpflow.likelihoods.Likelihood.variational_expectations(l, self.Fmu, self.Fvar, y)
                # compile and run the functions:
                F1 = sess.run(F1, feed_dict={x: x_data})
                F2 = sess.run(F2, feed_dict={x: x_data})
                self.assertTrue(np.allclose(F1, F2, test_setup.tolerance, test_setup.tolerance))

    def test_pred_density(self):
        # get all the likelihoods where predict_density  has been overwritten.
        with self.test_session() as sess:
            for test_setup in self.test_setups:
                if not test_setup.is_analytic:
                    continue
                l = test_setup.likelihood
                y = test_setup.Y
                x_data = l.get_free_state()
                # make parameters if needed
                x = tf.placeholder(float_type)
                l.make_tf_array(x)
                # 'build' the functions
                with l.tf_mode():
                    F1 = l.predict_density(self.Fmu, self.Fvar, y)
                    F2 = gpflow.likelihoods.Likelihood.predict_density(l, self.Fmu, self.Fvar, y)
                # compile and run the functions:
                F1 = sess.run(F1, feed_dict={x: x_data})
                F2 = sess.run(F2, feed_dict={x: x_data})
                self.assertTrue(np.allclose(F1, F2, test_setup.tolerance, test_setup.tolerance))


class TestRobustMaxMulticlass(GPflowTestCase):
    """
    Some specialized tests to the multiclass likelihood with RobustMax inverse link function.
    """
    def testSymmetric(self):
        """
        This test is based on the observation that for
        symmetric inputs the class predictions must have equal probability.
        """
        with self.test_session() as sess:
            nClasses = 5
            nPoints = 10
            tolerance = 1e-4
            epsilon = 1e-3
            F = tf.placeholder(float_type)
            x = tf.placeholder(float_type)
            F_data = np.ones((nPoints, nClasses))
            l = gpflow.likelihoods.MultiClass(nClasses)
            l.invlink.epsilon = epsilon
            rng = np.random.RandomState(1)
            Y = rng.randint(nClasses, size=(nPoints, 1))
            with l.tf_mode():
                mu, _ = sess.run(
                    l.predict_mean_and_var(F, F),
                    feed_dict={x: l.get_free_state(), F: F_data})
                pred = sess.run(
                    l.predict_density(F, F, Y),
                    feed_dict={x: l.get_free_state(), F: F_data})
                variational_expectations = sess.run(
                    l.variational_expectations(F, F, Y),
                    feed_dict={x: l.get_free_state(), F: F_data})
            expected_mu = (
                (1./nClasses * (1. - epsilon) +
                (1. - 1. / nClasses) * epsilon / (nClasses - 1)) *
                np.ones((nPoints, 1)))
            self.assertTrue(np.allclose(mu, expected_mu, tolerance, tolerance))
            expected_log_denisty = np.log(expected_mu)
            self.assertTrue(np.allclose(pred, expected_log_denisty, 1e-3, 1e-3))
            validation_variational_expectation = 1./nClasses * np.log(1. - epsilon) + \
                (1. - 1./nClasses) * np.log(epsilon / (nClasses - 1))
            self.assertTrue(
                np.allclose(
                    variational_expectations,
                    np.ones((nPoints, 1)) * validation_variational_expectation,
                    tolerance, tolerance))

    def testPredictDensity(self):
        tol = 1e-4
        num_points = 100
        mock_prob = 0.73

        class MockRobustMax(gpflow.likelihoods.RobustMax):
            def prob_is_largest(self, Y, Fmu, Fvar, gh_x, gh_w):
                return tf.ones((num_points, 1)) * mock_prob

        with self.test_session() as sess:
            epsilon = 0.231
            num_classes = 5
            l = gpflow.likelihoods.MultiClass(
                num_classes, invlink=MockRobustMax(num_classes, epsilon))

            F = tf.placeholder(float_type)
            y = tf.placeholder(float_type)
            F_data = np.ones((num_points, num_classes))
            rng = np.random.RandomState(1)
            Y_data = rng.randint(num_classes, size=(num_points, 1))

            with l.tf_mode():
                pred = sess.run(
                    l.predict_density(F, F, y),
                    feed_dict={F: F_data, y: Y_data})
            expected_prediction = -0.5499780059
            #^ evaluated on calculator: log( (1-\epsilon) * 0.73 + (1-0.73) * \epsilon/(num_classes -1))

            self.assertTrue(np.allclose(pred, expected_prediction, tol, tol))


class TestMulticlassIndexFix(GPflowTestCase):
    """
    A regression test for a bug in multiclass likelihood.
    """
    def testA(self):
        with self.test_session() as sess:
            mu, var = tf.placeholder(float_type), tf.placeholder(float_type)
            Y = tf.placeholder(tf.int32)
            lik = gpflow.likelihoods.MultiClass(3)
            ve = lik.variational_expectations(mu, var, Y)
            tf.gradients(tf.reduce_sum(ve), mu)


class TestSwitchedLikelihood(GPflowTestCase):
    """
    SwitchedLikelihood is saparately tested here.
    Here, we make sure the partition-stictch works fine.
    """
    def setUp(self):
        with self.test_session() as sess:
            rng = np.random.RandomState(1)
            self.Y_list = [rng.randn(3, 2),  rng.randn(4, 2),  rng.randn(5, 2)]
            self.F_list = [rng.randn(3, 2),  rng.randn(4, 2),  rng.randn(5, 2)]
            self.Fvar_list = [np.exp(rng.randn(3, 2)),  np.exp(rng.randn(4, 2)),  np.exp(rng.randn(5, 2))]
            self.Y_label = [np.ones((3, 1))*0, np.ones((4, 1))*1, np.ones((5, 1))*2]
            self.Y_perm = list(range(3+4+5))
            rng.shuffle(self.Y_perm)

            # shuffle the original data
            self.Y_sw = np.hstack([np.concatenate(self.Y_list), np.concatenate(self.Y_label)])[self.Y_perm, :]
            self.F_sw = np.concatenate(self.F_list)[self.Y_perm, :]
            self.Fvar_sw = np.concatenate(self.Fvar_list)[self.Y_perm, :]
            # likelihoods
            self.likelihoods = [gpflow.likelihoods.Gaussian(),
                                gpflow.likelihoods.Gaussian(),
                                gpflow.likelihoods.Gaussian()]
            for lik in self.likelihoods:
                lik.variance = np.exp(rng.randn(1))
            self.switched_likelihood = gpflow.likelihoods.SwitchedLikelihood(self.likelihoods)

            # initialize switched likelihood
            free_array = self.switched_likelihood.get_free_state()
            self.switched_likelihood.make_tf_array(free_array)
            sess.run(tf.global_variables_initializer())

    def test_logp(self):
        # switchedlikelihood
        with self.test_session() as sess, self.switched_likelihood.tf_mode():
            switched_rslt = sess.run(self.switched_likelihood.logp(self.F_sw, self.Y_sw))
            # likelihood
            rslts = []
            for lik, y, f in zip(self.likelihoods, self.Y_list, self.F_list):
                with lik.tf_mode():
                    rslts.append(sess.run(lik.logp(f, y)))

            self.assertTrue(np.allclose(switched_rslt, np.concatenate(rslts)[self.Y_perm, :]))

    def test_predict_density(self):
        with self.test_session() as sess, self.switched_likelihood.tf_mode():
            # switchedlikelihood
            switched_rslt = sess.run(
                self.switched_likelihood.predict_density(self.F_sw, self.Fvar_sw, self.Y_sw))
            # likelihood
            rslts = []
            for lik, y, f, fvar in zip(self.likelihoods,
                                       self.Y_list,
                                       self.F_list,
                                       self.Fvar_list):
                sess.run(tf.global_variables_initializer())
                with lik.tf_mode():
                    rslts.append(sess.run(lik.predict_density(f, fvar, y)))

            self.assertTrue(np.allclose(switched_rslt, np.concatenate(rslts)[self.Y_perm, :]))

    def test_variational_expectations(self):
        # switchedlikelihood
        with self.test_session() as sess, self.switched_likelihood.tf_mode():
            switched_rslt = sess.run(
                self.switched_likelihood.variational_expectations(
                    self.F_sw, self.Fvar_sw, self.Y_sw))
            # likelihood
            rslts = []
            for lik, y, f, fvar in zip(self.likelihoods,
                                       self.Y_list,
                                       self.F_list,
                                       self.Fvar_list):
                sess.run(tf.global_variables_initializer())
                with lik.tf_mode():
                    rslts.append(sess.run(lik.variational_expectations(f, fvar, y)))
            self.assertTrue(np.allclose(switched_rslt, np.concatenate(rslts)[self.Y_perm, :]))


class TestLikelihoodChecks(GPflowTestCase):
    def run_models(self, likelihood, Y):
        X = np.random.randn(Y.shape[0], 1)
        gpflow.gpr.GPR(X, Y, gpflow.kernels.RBF(1))
        m1 = gpflow.vgp.VGP(X, Y, gpflow.kernels.RBF(1), likelihood)
        m2 = gpflow.svgp.SVGP(X, Y, gpflow.kernels.RBF(1), likelihood, X, minibatch_size=1)

    def test_likelihood_checks(self):
        to_pass = [
            [gpflow.likelihoods.Gaussian(), np.array((1.)).reshape(1, 1)],
            [gpflow.likelihoods.Poisson(), np.array((1., 1., 3.)).reshape(3, 1)],
            [gpflow.likelihoods.Exponential(), np.array((1e-12, 1)).reshape(2, 1)],
            [gpflow.likelihoods.StudentT(), np.array((-1e-12, 1)).reshape(2, 1)],
            [gpflow.likelihoods.Bernoulli(), np.array((0., 1.)).reshape(2, 1)],
            [gpflow.likelihoods.Bernoulli(), np.array((-1., 1.)).reshape(2, 1)],
            [gpflow.likelihoods.Gamma(), np.array((1e-12, 1)).reshape(2, 1)],
            [gpflow.likelihoods.Beta(), np.array((1e-12, 1.)).reshape(2, 1)],
            [gpflow.likelihoods.MultiClass(3), np.array((0., 2.)).reshape(2, 1)],
            [gpflow.likelihoods.Ordinal(np.array((1., 2.))), np.array((0., 2.)).reshape(2, 1)],
        ]

        to_fail = [
            [gpflow.likelihoods.Gaussian(), np.array((1.)).reshape(1, 1, 1)],
            [gpflow.likelihoods.Gaussian(), np.array((1)).reshape(1, 1)],
            [gpflow.likelihoods.Poisson(), np.array((1.1)).reshape(1, 1)],
            [gpflow.likelihoods.Poisson(), np.array((-1.)).reshape(1, 1)],
            [gpflow.likelihoods.Exponential(), np.array((-1e-12, 1)).reshape(2, 1)],
            [gpflow.likelihoods.Gamma(), np.array((-1e-12, 1)).reshape(2, 1)],
            [gpflow.likelihoods.Beta(), np.array((-1e-12, 1.)).reshape(2, 1)],
            [gpflow.likelihoods.Beta(), np.array((1e-12, 1.1)).reshape(2, 1)],
            [gpflow.likelihoods.MultiClass(3), np.array((0.1, 2.)).reshape(2, 1)],
            [gpflow.likelihoods.MultiClass(3), np.array((1., 2.)).reshape(1, 2)],
            [gpflow.likelihoods.MultiClass(3), np.array((1., 3.)).reshape(2, 1)],
        ]

        to_warn = [
            [gpflow.likelihoods.Bernoulli(), np.array((2., 1., 0.)).reshape(3, 1)],
            [gpflow.likelihoods.Bernoulli(), np.array((2., 1.1)).reshape(2, 1)],
        ]

        # special case of switched likelihood
        sl = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gamma(),
                                                    gpflow.likelihoods.Gaussian()])
        A = np.array(((0, 1), (0, 1), (2, 0.))).reshape(3, 2)
        B = np.array(((0, 1), (0, 1), (2, 3.))).reshape(3, 2)
        to_pass.append([sl, A])
        to_fail.append([sl, B])

        for l, v in to_pass:
            with self.test_session(graph=tf.Graph()):
                self.run_models(l, v)

        for l, v, in to_fail:
            with self.test_session(graph=tf.Graph()):
                with self.assertRaises(ValueError):
                    self.run_models(l, v)

        for l, v, in to_warn:
            with self.test_session(graph=tf.Graph()):
                with self.assertRaises(Warning):
                    self.run_models(l, v)



if __name__ == "__main__":
    unittest.main()
