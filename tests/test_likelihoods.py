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

import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow import settings
from gpflow.test_util import GPflowTestCase


class LikelihoodSetup(object):
    def __init__(self, likelihood, Y, tolerance):
        self.likelihood, self.Y, self.tolerance = likelihood, Y, tolerance
        self.is_analytic = likelihood.predict_density is not \
                           gpflow.likelihoods.Likelihood.predict_density


def getLikelihoodSetups(includeMultiClass=True, addNonStandardLinks=False):
    test_setups = []
    rng = np.random.RandomState(1)
    for likelihoodClass in gpflow.likelihoods.Likelihood.__subclasses__():
        if likelihoodClass == gpflow.likelihoods.MonteCarloLikelihood:
            continue  # abstract base class
        if likelihoodClass == gpflow.likelihoods.Ordinal:
            test_setups.append(
                LikelihoodSetup(likelihoodClass(np.array([-1, 1])),
                                rng.randint(0, 3, (10, 2)), 1e-6))
        elif likelihoodClass == gpflow.likelihoods.SwitchedLikelihood:
            continue  # switched likelihood tested separately
        elif likelihoodClass == gpflow.likelihoods.MultiClass:
            if includeMultiClass:
                sample = rng.randn(10, 2)
                # Multiclass needs a less tight tolerance due to presence of clipping.
                tolerance = 1e-3
                test_setups.append(
                    LikelihoodSetup(likelihoodClass(2),
                                    np.argmax(sample, 1).reshape(-1, 1), tolerance))
        else:
            # most likelihoods follow this standard:
            test_setups.append(
                LikelihoodSetup(likelihoodClass(),
                                rng.rand(10, 2).astype(settings.float_type), 1e-6))

    if addNonStandardLinks:
        test_setups.append(LikelihoodSetup(gpflow.likelihoods.Poisson(invlink=tf.square),
                                           rng.rand(10, 2).astype(settings.float_type), 1e-6))
        test_setups.append(LikelihoodSetup(gpflow.likelihoods.Exponential(invlink=tf.square),
                                           rng.rand(10, 2).astype(settings.float_type), 1e-6))
        test_setups.append(LikelihoodSetup(gpflow.likelihoods.Gamma(invlink=tf.square),
                                           rng.rand(10, 2).astype(settings.float_type), 1e-6))

        def sigmoid(x):
            return 1. / (1 + tf.exp(-x))

        test_setups.append(LikelihoodSetup(gpflow.likelihoods.Bernoulli(invlink=sigmoid),
                                           rng.rand(10, 2).astype(settings.float_type), 1e-6))
    return test_setups


class TestPredictConditional(GPflowTestCase):
    """
    Here we make sure that the conditional_mean and contitional_var functions
    give the same result as the predict_mean_and_var function if the prediction
    has no uncertainty.
    """

    def setUp(self):
        self.test_graph = tf.Graph()

    def prepare(self):
        test_setups = getLikelihoodSetups(addNonStandardLinks=True)
        rng = np.random.RandomState(0)
        F = tf.placeholder(settings.float_type)
        F_data = rng.randn(10, 2).astype(settings.float_type)
        feed = {F: F_data}
        return test_setups, F, feed

    def test_mean(self):
        with self.test_context() as session:
            test_setups, F, feed = self.prepare()
            for test_setup in test_setups:
                l = test_setup.likelihood
                l.compile()
                mu1 = session.run(l.conditional_mean(F), feed_dict=feed)
                zero = F * 0.
                mu2, _ = session.run(l.predict_mean_and_var(F, zero), feed_dict=feed)
                assert_allclose(mu1, mu2, test_setup.tolerance, test_setup.tolerance)

    def test_variance(self):
        with self.test_context() as session:
            test_setups, F, feed = self.prepare()
            for test_setup in test_setups:
                l = test_setup.likelihood
                l.compile()
                zero = F * 0.
                v1 = session.run(l.conditional_variance(F), feed_dict=feed)
                v2 = session.run(l.predict_mean_and_var(F, zero)[1], feed_dict=feed)
                assert_allclose(v1, v2, atol=test_setup.tolerance)

    def test_var_exp(self):
        """
        Here we make sure that the variational_expectations gives the same result
        as logp if the latent function has no uncertainty.
        """
        with self.test_context() as session:
            test_setups, F, feed = self.prepare()
            for test_setup in test_setups:
                l = test_setup.likelihood
                y = test_setup.Y
                l.compile()
                r1 = session.run(l.logp(F, y), feed_dict=feed)
                zero = F * 0.
                r2 = session.run(
                    l.variational_expectations(F, zero, test_setup.Y), feed_dict=feed)
                assert_allclose(r1, r2, atol=test_setup.tolerance, rtol=test_setup.tolerance)


class TestQuadrature(GPflowTestCase):
    """
    Where quadrature methods have been overwritten, make sure the new code
    does something close to the quadrature
    """

    def setUp(self):
        self.test_graph = tf.Graph()
        self.rng = np.random.RandomState()
        self.Fmu, self.Fvar, self.Y = self.rng.randn(3, 10, 2).astype(settings.float_type)
        self.Fvar = 0.01 * (self.Fvar ** 2)
        with self.test_context():
            self.test_setups = getLikelihoodSetups(includeMultiClass=False)

    def test_var_exp(self):
        for test_setup in self.test_setups:
            with self.test_context() as session:
                # get all the likelihoods where variational expectations has been overwritten
                if not test_setup.is_analytic:
                    continue
                l = test_setup.likelihood
                y = test_setup.Y
                # 'build' the functions
                l.compile()
                F1 = l.variational_expectations(self.Fmu, self.Fvar, y)
                F2 = gpflow.likelihoods.Likelihood.variational_expectations(
                    l, self.Fmu, self.Fvar, y)
                # compile and run the functions:
                F1 = session.run(F1)
                F2 = session.run(F2)
                assert_allclose(F1, F2, test_setup.tolerance, test_setup.tolerance)

    def test_pred_density(self):
        # get all the likelihoods where predict_density has been overwritten.
        for test_setup in self.test_setups:
            with self.test_context() as session:
                if not test_setup.is_analytic:
                    continue
                l = test_setup.likelihood
                y = test_setup.Y
                l.compile()
                # 'build' the functions
                F1 = l.predict_density(self.Fmu, self.Fvar, y)
                F2 = gpflow.likelihoods.Likelihood.predict_density(l, self.Fmu, self.Fvar, y)
                # compile and run the functions:
                F1 = session.run(F1)
                F2 = session.run(F2)
                assert_allclose(F1, F2, test_setup.tolerance, test_setup.tolerance)

    def test_pred_mean_and_var(self):
        # get all the likelihoods where predict_density has been overwritten.
        for test_setup in self.test_setups:
            with self.test_context() as session:
                if not test_setup.is_analytic:
                    continue
                l = test_setup.likelihood
                l.compile()
                # 'build' the functions
                F1 = l.predict_mean_and_var(self.Fmu, self.Fvar)
                F2 = gpflow.likelihoods.Likelihood.predict_mean_and_var(l, self.Fmu, self.Fvar)
                # compile and run the functions:
                F1 = session.run(F1)
                F2 = session.run(F2)
                assert_allclose(F1, F2, test_setup.tolerance, test_setup.tolerance)


class TestMonteCarlo(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        self.rng = np.random.RandomState()
        self.rng.seed(1)
        self.Fmu, self.Fvar, self.Y = self.rng.randn(3, 10, 1).astype(settings.float_type)
        self.Fvar = 0.01 * (self.Fvar ** 2)

    def test_var_exp(self):
        with self.test_context() as session:
            tf.set_random_seed(1)
            l = gpflow.likelihoods.GaussianMC(0.3)
            l.num_monte_carlo_points = 1000000
            # 'build' the functions
            l.compile()
            F1 = l.variational_expectations(self.Fmu, self.Fvar, self.Y)
            F2 = gpflow.likelihoods.Gaussian.variational_expectations(
                l, self.Fmu, self.Fvar, self.Y)
            # compile and run the functions:
            F1 = session.run(F1)
            F2 = session.run(F2)
            assert_allclose(F1, F2, rtol=5e-4, atol=1e-4)

    def test_pred_density(self):
        with self.test_context() as session:
            tf.set_random_seed(1)
            l = gpflow.likelihoods.GaussianMC(0.3)
            l.num_monte_carlo_points = 1000000
            l.compile()
            # 'build' the functions
            F1 = l.predict_density(self.Fmu, self.Fvar, self.Y)
            F2 = gpflow.likelihoods.Gaussian.predict_density(l, self.Fmu, self.Fvar, self.Y)
            # compile and run the functions:
            F1 = session.run(F1)
            F2 = session.run(F2)
            assert_allclose(F1, F2, rtol=5e-4, atol=1e-4)

    def test_pred_mean_and_var(self):
        with self.test_context() as session:
            tf.set_random_seed(1)
            l = gpflow.likelihoods.GaussianMC(0.3)
            l.num_monte_carlo_points = 1000000
            l.compile()
            # 'build' the functions
            F1 = l.predict_mean_and_var(self.Fmu, self.Fvar)
            F2 = gpflow.likelihoods.Gaussian.predict_mean_and_var(l, self.Fmu, self.Fvar)
            # compile and run the functions:
            F1m, F1v = session.run(F1)
            F2m, F2v = session.run(F2)
            assert_allclose(F1m, F2m, rtol=5e-4, atol=1e-4)
            assert_allclose(F1v, F2v, rtol=5e-4, atol=1e-4)


class TestSoftMax(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        self.rng = np.random.RandomState(1)

    def prepare(self, dimF, dimY, num=10):
        feed = {}

        def make_tensor(data, dtype=settings.float_type):
            tensor = tf.placeholder(dtype)
            feed[tensor] = data.astype(dtype)
            return tensor

        dF = np.vstack((self.rng.randn(num - 3, dimF), np.array([[-3., 0.], [3, 0.], [0., 0.]]))) if dimF == 2 else \
            self.rng.randn(num, dimF)
        dY = np.vstack((self.rng.randn(num - 3, dimY), np.ones((3, dimY)))) > 0
        F = make_tensor(dF)
        Y = make_tensor(dY, settings.int_type)  # 0 or 1
        return F, Y, feed

    def test_y_shape_assert(self):
        """
        SoftMax assumes the class is given as a label (not, e.g., one-hot
        encoded), and hence just uses the first column of Y. To prevent
        silent errors, there is a tf assertion that ensures Y only has one
        dimension. This test checks that this assert works as intended.
        """
        with self.test_context() as sess:
            F, Y, feed = self.prepare(dimF=5, dimY=2)
            l = gpflow.likelihoods.SoftMax(5)
            l.compile()
            try:
                sess.run(l.logp(F, Y), feed_dict=feed)
            except tf.errors.InvalidArgumentError as e:
                assert "assertion failed" in e.message

    def test_bernoulli_equivalence(self):
        with self.test_context() as sess:
            F, Y, feed = self.prepare(dimF=2, dimY=1)
            Fvar = tf.exp(tf.stack([F[:, 1], -10.0 + tf.zeros(tf.shape(F)[0], dtype=F.dtype)], axis=1))
            F = tf.stack([F[:, 0], tf.zeros(tf.shape(F)[0], dtype=F.dtype)], axis=1)
            Ylabel = 1 - Y  # We need the 1 - Y, as we need to pass the *label* to SoftMax

            def logistic_link(x):
                return 1.0 / (1.0 + tf.exp(-x))

            ls = gpflow.likelihoods.SoftMax(2)
            ls.num_monte_carlo_points = 10000000
            ls.compile()
            lb = gpflow.likelihoods.Bernoulli(invlink=logistic_link)
            lb.num_gauss_hermite_points = 50
            lb.compile()

            ls_cm = sess.run(ls.conditional_mean(F), feed_dict=feed)[:, :1]
            lb_cm = sess.run(lb.conditional_mean(F[:, :1]), feed_dict=feed)
            ls_cv = sess.run(ls.conditional_variance(F), feed_dict=feed)[:, :1]
            lb_cv = sess.run(lb.conditional_variance(F[:, :1]), feed_dict=feed)
            ls_lp = sess.run(ls.logp(F, Ylabel), feed_dict=feed)
            lb_lp = sess.run(lb.logp(F[:, :1], Y), feed_dict=feed)

            assert_allclose(ls_cm, lb_cm)
            assert_allclose(ls_cv, lb_cv)
            assert_allclose(ls_lp, lb_lp)

            ls_pm, ls_pv = sess.run(ls.predict_mean_and_var(F, Fvar), feed_dict=feed)
            lb_pm, lb_pv = sess.run(lb.predict_mean_and_var(F[:, :1], Fvar[:, :1]), feed_dict=feed)

            assert_allclose(ls_pm[:, 0, None], lb_pm, rtol=1e-3)
            assert_allclose(ls_pv[:, 0, None], lb_pv, rtol=1e-3)

            ls_ve = sess.run(ls.variational_expectations(F, Fvar, Ylabel), feed_dict=feed)
            lb_ve = sess.run(lb.variational_expectations(F[:, :1], Fvar[:, :1], Y), feed_dict=feed)

            assert_allclose(ls_ve[:, 0, None], lb_ve, rtol=1e-3)


class TestRobustMaxMulticlass(GPflowTestCase):
    """
    Some specialized tests to the multiclass likelihood with RobustMax inverse link function.
    """

    def setUp(self):
        self.test_graph = tf.Graph()

    def testSymmetric(self):
        """
        This test is based on the observation that for
        symmetric inputs the class predictions must have equal probability.
        """
        with self.test_context() as session:
            nClasses = 5
            nPoints = 10
            tolerance = 1e-4
            epsilon = 1e-3
            F = tf.placeholder(settings.float_type)
            F_data = np.ones((nPoints, nClasses))
            feed = {F: F_data}
            rng = np.random.RandomState(1)
            Y = rng.randint(nClasses, size=(nPoints, 1))

            l = gpflow.likelihoods.MultiClass(nClasses)
            l.invlink.epsilon = epsilon
            l.compile()

            mu, _ = session.run(l.predict_mean_and_var(F, F), feed_dict=feed)
            pred = session.run(l.predict_density(F, F, Y), feed_dict=feed)
            variational_expectations = session.run(
                l.variational_expectations(F, F, Y), feed_dict=feed)
            expected_mu = (1. / nClasses * (1. - epsilon) + (1. - 1. / nClasses) * \
                           epsilon / (nClasses - 1)) * np.ones((nPoints, 1))

            self.assertTrue(np.allclose(mu, expected_mu, tolerance,
                                        tolerance))  # assert_allclose() would complain about shape mismatch
            expected_log_denisty = np.log(expected_mu)
            self.assertTrue(np.allclose(pred, expected_log_denisty, 1e-3, 1e-3))
            validation_variational_expectation = 1. / nClasses * np.log(1. - epsilon) + \
                                                 (1. - 1. / nClasses) * np.log(epsilon / (nClasses - 1))
            assert_allclose(
                variational_expectations,
                np.ones((nPoints, 1)) * validation_variational_expectation,
                tolerance, tolerance)

    def testPredictDensity(self):
        tol = 1e-4
        num_points = 100
        mock_prob = 0.73

        class MockRobustMax(gpflow.likelihoods.RobustMax):
            def prob_is_largest(self, Y, Fmu, Fvar, gh_x, gh_w):
                return tf.ones((num_points, 1), dtype=settings.float_type) * mock_prob

        with self.test_context() as session:
            epsilon = 0.231
            num_classes = 5
            l = gpflow.likelihoods.MultiClass(
                num_classes, invlink=MockRobustMax(num_classes, epsilon))
            l.compile()

            F = tf.placeholder(settings.float_type)
            y = tf.placeholder(settings.float_type)
            F_data = np.ones((num_points, num_classes))
            rng = np.random.RandomState(1)
            Y_data = rng.randint(num_classes, size=(num_points, 1))
            feed = {F: F_data, y: Y_data}

            pred = session.run(l.predict_density(F, F, y), feed_dict=feed)
            expected_prediction = -0.5499780059
            # ^^^ evaluated on calculator:
            # log((1-\epsilon) * 0.73 + (1-0.73) * \epsilon/(num_classes -1))

            assert_allclose(pred, expected_prediction, tol, tol)

    def testEpsK1Changes(self):
        """
        Checks that eps K1 changes when epsilon changes. This used to not happen and had to be manually changed.
        """
        with self.test_context() as session:
            initial_eps = 1e-3
            num_classes = 5
            rm = gpflow.likelihoods.RobustMax(num_classes, initial_eps)

            expected_eps_k1 = initial_eps / (num_classes - 1.)
            actual_eps_k1 = session.run(rm._eps_K1)
            self.assertAlmostEqual(expected_eps_k1, actual_eps_k1)

            new_eps = 0.412
            rm.epsilon.assign(new_eps, session=session)
            expected_eps_k2 = new_eps / (num_classes - 1.)
            actual_eps_k2 = session.run(rm._eps_K1)
            self.assertAlmostEqual(expected_eps_k2, actual_eps_k2)


class TestMulticlassIndexFix(GPflowTestCase):
    """
    A regression test for a bug in multiclass likelihood.
    """

    def testA(self):
        with self.test_context():
            mu = tf.placeholder(settings.float_type)
            var = tf.placeholder(settings.float_type)
            Y = tf.placeholder(tf.int32)
            lik = gpflow.likelihoods.MultiClass(3)
            ve = lik.variational_expectations(mu, var, Y)
            tf.gradients(tf.reduce_sum(ve), mu)


class TestSwitchedLikelihood(GPflowTestCase):
    """
    SwitchedLikelihood is separately tested here.
    Here, we make sure the partition-stitch works fine.
    """

    def setUp(self):
        self.test_graph = tf.Graph()

        with self.test_context():
            rng = np.random.RandomState(1)
            self.Y_list = [rng.randn(3, 2), rng.randn(4, 2), rng.randn(5, 2)]
            self.F_list = [rng.randn(3, 2), rng.randn(4, 2), rng.randn(5, 2)]
            self.Fvar_list = [np.exp(rng.randn(3, 2)), np.exp(rng.randn(4, 2)),
                              np.exp(rng.randn(5, 2))]
            self.Y_label = [np.ones((3, 1)) * 0, np.ones((4, 1)) * 1, np.ones((5, 1)) * 2]
            self.Y_perm = list(range(3 + 4 + 5))
            rng.shuffle(self.Y_perm)

            # shuffle the original data
            self.Y_sw = np.hstack([
                np.concatenate(self.Y_list),
                np.concatenate(self.Y_label)])[self.Y_perm, :]
            self.F_sw = np.concatenate(self.F_list)[self.Y_perm, :]
            self.Fvar_sw = np.concatenate(self.Fvar_list)[self.Y_perm, :]
            # likelihoods
            self.likelihoods = [gpflow.likelihoods.Gaussian(),
                                gpflow.likelihoods.Gaussian(),
                                gpflow.likelihoods.Gaussian()]
            for lik in self.likelihoods:
                lik.variance = np.exp(rng.randn(1)).squeeze()
            self.switched_likelihood = gpflow.likelihoods.SwitchedLikelihood(self.likelihoods)

    def test_logp(self):
        # switchedlikelihood
        with self.test_context() as session:
            self.switched_likelihood.compile()
            switched_rslt = session.run(self.switched_likelihood.logp(self.F_sw, self.Y_sw))
            rslts = []
            for lik, y, f in zip(self.likelihoods, self.Y_list, self.F_list):
                rslts.append(session.run(lik.logp(f, y)))
            assert_allclose(switched_rslt, np.concatenate(rslts)[self.Y_perm, :])

    def test_predict_density(self):
        with self.test_context() as session:
            self.switched_likelihood.compile()
            # switchedlikelihood
            switched_rslt = session.run(
                self.switched_likelihood.predict_density(self.F_sw, self.Fvar_sw, self.Y_sw))
            # likelihood
            rslts = []
            for lik, y, f, fvar in zip(self.likelihoods,
                                       self.Y_list,
                                       self.F_list,
                                       self.Fvar_list):
                rslts.append(session.run(lik.predict_density(f, fvar, y)))
            assert_allclose(switched_rslt, np.concatenate(rslts)[self.Y_perm, :])

    def test_variational_expectations(self):
        # switchedlikelihood
        with self.test_context() as session:
            self.switched_likelihood.compile()
            switched_rslt = session.run(
                self.switched_likelihood.variational_expectations(
                    self.F_sw, self.Fvar_sw, self.Y_sw))
            rslts = []
            for lik, y, f, fvar in zip(self.likelihoods,
                                       self.Y_list,
                                       self.F_list,
                                       self.Fvar_list):
                rslts.append(session.run(lik.variational_expectations(f, fvar, y)))
            assert_allclose(switched_rslt, np.concatenate(rslts)[self.Y_perm, :])


class TestSwitchedLikelihoodRegression(GPflowTestCase):
    """
    A Regression test when using Switched likelihood: the number of latent
    functions in a GP model must be equal to the number of columns in Y minus
    one. The final column of Y is used to index the switch. If the number of
    latent functions does not match, an exception will be raised.
    """

    def setUp(self):
        self.test_graph = tf.Graph()

        with self.test_context():
            rng = np.random.RandomState(1)
            self.X = rng.rand(100, 1)
            self.Y = np.hstack((np.random.randn(100, 1), np.random.randint(0, 3, (100, 1))))
            self.likelihoods = [gpflow.likelihoods.StudentT(),
                                gpflow.likelihoods.StudentT(),
                                gpflow.likelihoods.StudentT()]
            self.switched_likelihood = gpflow.likelihoods.SwitchedLikelihood(self.likelihoods)

    def test_correct_num_latent(self):
        with self.test_context():
            m = gpflow.models.VGP(self.X, self.Y, kern=gpflow.kernels.Matern12(1),
                                  likelihood=self.switched_likelihood, num_latent=1)
            m.compute_log_likelihood()  # should compute something!

    def test_bad_num_latent(self):
        with self.test_context():
            m = gpflow.models.VGP(self.X, self.Y, kern=gpflow.kernels.Matern12(1),
                                  likelihood=self.switched_likelihood, num_latent=2)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                m.compute_log_likelihood()  # should die


if __name__ == "__main__":
    tf.test.main()
