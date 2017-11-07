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


class BaseTestMomentFunctions(GPflowTestCase):
    def _importance_sampled_moments_test_runner(self, rng, num_samples, rtol, nus, taus,
                                                other_params_arrs, weight_func, tf_moments_cls):

        #other_params_arrs should be in the correct order for tf_moments_cls
        #and weight func should be written to respect this order too.

        # 1. Compute the moments using numpy/scipy using importance sampling
        zero_moments = []
        first_moments = []
        second_moments = []
        for nu, tau, *other_dist_args in zip(nus, taus, *other_params_arrs):
            mu = nu / tau
            sigma_sq = 1. / tau
            normal_mi = stats.norm(mu, np.sqrt(sigma_sq))
            samples = normal_mi.rvs(size=(num_samples, 1), random_state=rng)

            weights = weight_func(samples, *other_dist_args)
            estimated_z = np.mean(weights, keepdims=True)

            z_prime = estimated_z
            zero_moments.append(z_prime)

            first_mom = np.mean(weights * samples, keepdims=True) / z_prime
            first_moments.append(first_mom)
            expectation_x_squared = np.mean(weights * samples ** 2, keepdims=True) / z_prime
            second_mom = expectation_x_squared - first_mom ** 2
            second_moments.append(second_mom)

        log_zero_moments_via_scipy = np.log(np.concatenate(zero_moments))
        first_moments_via_scipy = np.concatenate(first_moments)
        second_moments_via_scipy = np.concatenate(second_moments)

        # 2. Compute the moments via the analyticial TF expressions
        nu_tf = tf.placeholder(tf.float64, shape=(3, 1))
        tau_tf = tf.placeholder(tf.float64, shape=(3, 1))
        log_zeroth_moment_tf = tf_moments_cls(*other_params_arrs).calc_log_zero_moment(
            tau_tf, nu_tf)

        first_moment_tf, second_moment_tf \
            = tf_moments_cls(*other_params_arrs)._calc_first_and_second_centmoments(tau_tf, nu_tf)

        with tf.Session() as sess:
            log_zero_moment_via_tf_evald, first_moment_tf_evald, second_moment_tf_evald = \
                sess.run([log_zeroth_moment_tf, first_moment_tf, second_moment_tf], feed_dict={
                    nu_tf: nus, tau_tf: taus
                })

        # 3. Check that the two compute routes result in the same 'rough' answer,
        self.assertTrue(
            np.allclose(log_zero_moments_via_scipy, log_zero_moment_via_tf_evald, rtol=rtol),
            msg="log zero moments do not match (Scipy {} vs TF {})".format(
                log_zero_moments_via_scipy, log_zero_moment_via_tf_evald))
        self.assertTrue(np.allclose(first_moment_tf_evald, first_moments_via_scipy, rtol=rtol),
                        msg="first moments do not match(Scipy {} vs TF {})".format(
                            first_moments_via_scipy, first_moment_tf_evald
                        ))
        self.assertTrue(np.allclose(second_moments_via_scipy, second_moment_tf_evald, rtol=rtol),
                        msg="second moments do not match.")


class TestStepFunction(BaseTestMomentFunctions):
    def test_first_and_second_moments(self):
        """
        So we're gonna test both our implementation of the Upper thresholded Gaussian moment
        forumulae and the formulae itself by comparing against the emprical estimates from sampling
        from the distribution.
        """
        # Setup inputs
        means = np.array([0.2, -0.87, 0.451])[:, np.newaxis]
        vars = np.array([0.1, 0.32, 0.287])[:, np.newaxis]
        rng = np.random.RandomState(1000)
        NUM_SAMPLES = 100000
        rtol = 1e-2

        threshold = 0.544
        self._first_and_second_moment_tester(rng, NUM_SAMPLES, means, vars, threshold, True, rtol)

        threshold = -0.686
        self._first_and_second_moment_tester(rng, NUM_SAMPLES, means, vars, threshold, False, rtol)

    def test_log_zeroth_moment(self):
        # we will test this against a numpy/scipy implementation.
        means = np.array([0.2, -2.23, 0.451])[:, np.newaxis]
        variances = np.array([0.1, 0.32, 0.287])[:, np.newaxis]
        threshold = 0.3

        self._log_moment_tester(means, variances, threshold, True)
        self._log_moment_tester(means, variances, -0.89, False)

    def _first_and_second_moment_tester(self, rng, num_samples, means, vars,
                                                          threshold, start_high, rtol):

        # Do expected
        first_moments = []
        second_moments = []
        for mean, var in zip(means, vars):
            gaussian = stats.norm(loc=mean, scale=np.sqrt(var))
            samples = _draw_samples_from_upper_truncated_gaussian(gaussian, rng,
                num_samples=num_samples, threshold=threshold, truncation_of_higher_tail=start_high)
            first_moments.append(np.mean(samples))
            second_moments.append(np.var(samples))

        first_moment_via_scipy = np.array(first_moments)[:, None]
        second_moment_via_scipy = np.array(second_moments)[:, None]

        # Do it via tf
        means_tf = tf.placeholder(tf.float64)
        vars_tf = tf.placeholder(tf.float64)
        moments_tf = \
            gpflow.inference_helpers.StepFunction(threshold, start_high)\
                ._calc_first_and_second_centmoments(1. / vars_tf, (1. / vars_tf) * means_tf)

        with tf.Session() as sess:
            first_moment_via_tf, second_moment_via_tf = sess.run(moments_tf, feed_dict={
                means_tf: means, vars_tf:vars
            })

        self.assertTrue(np.allclose(first_moment_via_tf, first_moment_via_tf, rtol=rtol))
        self.assertTrue(np.allclose(second_moment_via_tf, second_moment_via_scipy, rtol=rtol))

    def _log_moment_tester(self, means, variances, threshold, start_high):

        # First do it via Scipy
        zero_moments = []
        for mu, var in zip(means, variances):
            if start_high:
                first_moment = stats.norm.logcdf(threshold, loc=mu, scale=np.sqrt(var))
            else:
                first_moment = np.log(1 - stats.norm.cdf(threshold, loc=mu, scale=np.sqrt(var)))
            zero_moments.append(first_moment)

        zero_moment_via_scipy = np.array(zero_moments)

        # Then do it via TF.
        means_tf = tf.placeholder(tf.float64)
        vars_tf = tf.placeholder(tf.float64)
        moments_tf = \
            gpflow.inference_helpers.StepFunction(
                threshold, start_high).calc_log_zero_moment(1. / vars_tf, (1. / vars_tf) * means_tf)

        with tf.Session() as sess:
            zero_moment_via_tf = sess.run(moments_tf, feed_dict={
                means_tf: means, vars_tf: variances})

        np.testing.assert_almost_equal(zero_moment_via_tf, zero_moment_via_scipy, decimal=6)


class TestUnormalisedGaussian(BaseTestMomentFunctions):
    def test_all_moments(self):
        # 1. Setup
        rng = np.random.RandomState(100)
        NUM_SAMPLES = 5000000
        mean_array = np.array([0.1, 1.6, 0.193])[:, None]
        std_array = np.array([0.86, 1.2, 2.43])[:, None]
        Zs_array = np.array([1.8, 0.23, 0.56])[:, None]

        nu_mi = np.array([0., -1.23, 0.451])[:, np.newaxis]
        tau_mi = np.array([1., 2.48, 0.287])[:, np.newaxis]

        def weight_func(samples, Z, mean, std):
            normal_mi = stats.norm(loc=mean, scale=std)
            return normal_mi.pdf(samples) * Z

        tf_moments_cls = gpflow.inference_helpers.UnormalisedGaussian
        self._importance_sampled_moments_test_runner(rng, NUM_SAMPLES, 1e-2, nu_mi, tau_mi,
                                                     [Zs_array, mean_array, std_array], weight_func, tf_moments_cls)


class TestCDFNormalGaussian(BaseTestMomentFunctions):
    def test_all_moments(self):
        # 1. Setup
        rng = np.random.RandomState(100)
        NUM_SAMPLES = 10000000
        m_array = np.array([0., 1.6, 0.193])[:, None]
        v_array = np.array([1., 1.2, 2.34])[:, None]

        nu_mi = np.array([0., -1.23, 0.451])[:, np.newaxis]
        tau_mi = np.array([1., 2.48, 0.287])[:, np.newaxis]

        def weight_func(samples, m, v):
            std_normal = stats.norm(0, 1.)
            weights = std_normal.cdf((samples - m) / v)
            return weights

        tf_moments_cls = gpflow.inference_helpers.CDFNormalGaussian
        self._importance_sampled_moments_test_runner(rng, NUM_SAMPLES, 1e-2, nu_mi, tau_mi,
                                                     [m_array, v_array], weight_func, tf_moments_cls)


def _draw_samples_from_upper_truncated_gaussian(gaussian, rng, num_samples, threshold, truncation_of_higher_tail):
    # sample from the Thresholded Gaussian distribution by importance sampling.
    BATCH_SIZE = int(1.5 * num_samples)
    completed_samples = np.array([], dtype=np.float64)
    while completed_samples.size < num_samples:
        samples = gaussian.rvs(size=BATCH_SIZE, random_state=rng)
        samples = samples[samples <= threshold] if truncation_of_higher_tail else samples[samples >= threshold]
        completed_samples = np.concatenate((completed_samples, samples))
    return completed_samples[:num_samples]
