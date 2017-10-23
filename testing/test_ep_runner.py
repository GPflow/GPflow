# Copyright 2017 John Bradshaw
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

from __future__ import print_function

from scipy import stats
import tensorflow as tf
import numpy as np

import gpflow
from gpflow.test_util import GPflowTestCase
from gpflow import settings

float_type = settings.dtypes.float_type


class TestEpRunner(GPflowTestCase):
    def test_compute_newest_sigma_and_mu(self):
        """
        Tests that we can find the mean and variance of a Gaussian multiplied by a Gaussian by
        checking against results found by importance sampling.
        """
        # 1. Setup the initial arrays
        NUM_SAMPLES = 1000000
        rng = np.random.RandomState(1000)
        mu_0 = np.array([0., 1.8])[:, None]
        var_0 = np.array([[2.3, 0.7], [0.7, 1.]])

        nu_1 = np.array([0.7, 0.21])[:, None]
        tau_1 = np.array([1., 0.1243])[:, None]

        # 2. we are going to compute the mean and variance empirically via importance sampling
        var_1 = np.diag(1./np.squeeze(tau_1))
        mu_1 = nu_1 / tau_1

        mvn_0 = stats.multivariate_normal(mean=np.squeeze(mu_0), cov=var_0)
        mvn_1 = stats.multivariate_normal(mean=np.squeeze(mu_1), cov=var_1)

        samples = mvn_0.rvs(size=NUM_SAMPLES, random_state=rng)
        weights = mvn_1.pdf(samples)[:, None]
        Z = np.mean(weights, axis=0, keepdims=True).T
        sampled_mean = np.mean(weights * samples, axis=0, keepdims=True).T / Z
        var_diag = np.mean(weights * (samples**2 - sampled_mean.T**2), axis=0, keepdims=True).T /Z
        covar = np.mean(weights * (np.product(samples, axis=1, keepdims=True) - np.product(sampled_mean)), keepdims=True) / Z
        expected_var = np.diag(np.squeeze(var_diag))
        expected_var[0,1] = covar
        expected_var[1, 0] = covar

        # 3. Then run the Tensorflow version
        ep_runner = gpflow.inference_helpers.EPRunner(
            lambda : gpflow.inference_helpers.UnormalisedGaussian(1., 0., 1.),
        )
        tau_tilde_ph = tf.placeholder(tf.float64)
        nu_tilde_ph = tf.placeholder(tf.float64)
        Sigma_0_ph = tf.placeholder(tf.float64)
        mu_0_ph = tf.placeholder(tf.float64)
        Sigma_new, mu_new = ep_runner.compute_newest_sigma_and_mu(tau_tilde_ph, nu_tilde_ph,
                                                                  Sigma_0_ph, mu_0_ph)

        with tf.Session() as sess:
            sigma_new_evald, mu_new_evald = sess.run([Sigma_new, mu_new], feed_dict={
                tau_tilde_ph: tau_1, nu_tilde_ph:nu_1, Sigma_0_ph:var_0, mu_0_ph:mu_0
            })

        # 4. Check the two match up (only roughly as the other is only an estimate)
        self.assertTrue(np.allclose(expected_var, sigma_new_evald, rtol=0.01),
                        msg="Variance matrices did not match up")
        self.assertTrue(np.allclose(mu_new_evald, sampled_mean, rtol=0.01),
                        msg="Means did not match up")

    def test_run_ep(self):
        """
        So we test EP works by making the distribution it is trying to approximate have the same
        form as its approximation factors. Therefore we expect the approximation factors to find the
        exact correct answer.
        """
        Z_array = np.array([[0.7, 1.2]]).T
        mu_array = np.array([[-0.1, 1.1]]).T
        sigma_array = np.array([[0.2, 0.7]]).T
        ep_runner = gpflow.inference_helpers.EPRunner(
            lambda :gpflow.inference_helpers.UnormalisedGaussian(Z_array, mu_array, sigma_array),
        )

        orig_mu = np.array([[-0.1, 1.5]]).T
        orig_var = np.array([[0.8, 0.1], [0.1, 0.9]])

        # Do Tensorflow section
        orig_mu_ph = tf.placeholder(tf.float64, shape=[2, 1])
        orig_var_ph = tf.placeholder(tf.float64, shape=[2, 2])
        nu_ph = tf.placeholder(tf.float64, shape=[2, 1])
        tau_ph = tf.placeholder(tf.float64, shape=[2, 1])

        Sigma_new, mu_new = ep_runner.compute_newest_sigma_and_mu(tau_ph, nu_ph,
                                                                  orig_var_ph, orig_mu_ph)

        with tf.Session() as sess:
            sigma_new_evald, mu_new_evald = sess.run([Sigma_new, mu_new], feed_dict={
                nu_ph:mu_array/sigma_array**2, tau_ph:1./sigma_array**2,
                orig_var_ph:orig_var, orig_mu_ph:orig_mu
            })

            ep_runner.compile(sess)
            ep_results = ep_runner.run_ep(orig_mu_ph, orig_var_ph, nu_ph, tau_ph)
            sigma_new_via_ep, mu_new_via_ep, found_tau = sess.run([ep_results.sigma,
                                                ep_results.mu, ep_results.new_tau_tilde],
                            feed_dict={tau_ph:np.zeros_like(mu_array, dtype=np.float64),
                                       nu_ph:np.zeros_like(mu_array, dtype=np.float64),
                orig_var_ph:orig_var, orig_mu_ph:orig_mu})

        np.testing.assert_almost_equal(1./sigma_array**2, found_tau)
        np.testing.assert_almost_equal(sigma_new_via_ep, sigma_new_evald)
        np.testing.assert_almost_equal(mu_new_via_ep, mu_new_evald)

    def test_calculate_log_normalising_constant(self):
        """
        Checks that it finds the correct log normalising constant by giving the distribution it is
        trying to approximate have the exact same form as the the approximation factors.
        We then estimate what the normaising constant should be via importance sampling and compare
        this to what we find here.
        """
        NUM_SAMPLES = 10000000
        rng = np.random.RandomState(100)

        Z_array = np.array([[1.7, 1.3]]).T
        mu_array = np.array([[-0.1, 1.1]]).T
        sigma_array = np.array([[0.2, 0.7]]).T
        ep_runner = gpflow.inference_helpers.EPRunner(
            lambda : gpflow.inference_helpers.UnormalisedGaussian(Z_array, mu_array, sigma_array),
        )

        orig_mu = np.array([[0.7, 0.9]]).T
        orig_var = np.array([[0.8, 0.1], [0.1, 0.9]])

        # Do the approximation
        mvn_0 = stats.multivariate_normal(mean=np.squeeze(orig_mu), cov=orig_var)
        var1 = np.diag(np.squeeze(sigma_array)**2)
        mvn_1 = stats.multivariate_normal(mean=np.squeeze(mu_array), cov=var1)

        samples = mvn_0.rvs(size=NUM_SAMPLES, random_state=rng)
        weights = mvn_1.pdf(samples)[:, None]
        Z = np.mean(weights, axis=0, keepdims=True).T * np.product(Z_array)

        # Do Tensorflow section
        orig_mu_ph = tf.placeholder(tf.float64, shape=[2, 1])
        orig_var_ph = tf.placeholder(tf.float64, shape=[2, 2])
        nu_ph = tf.placeholder(tf.float64, shape=[2, 1], name="nu_ph")
        tau_ph = tf.placeholder(tf.float64, shape=[2, 1], name="tau_ph")

        with tf.Session() as sess:
            lZ = ep_runner.calculate_log_normalising_constant(orig_mu_ph, orig_var_ph, nu_ph, tau_ph)
            lZ_evald = sess.run(lZ, feed_dict={tau_ph: 1./(sigma_array**2),
                nu_ph: mu_array/(sigma_array**2), orig_var_ph: orig_var, orig_mu_ph: orig_mu})

        self.assertTrue(np.allclose(lZ_evald, np.log(Z), rtol=1e-2))


