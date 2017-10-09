from __future__ import absolute_import, print_function

import unittest
import tensorflow as tf
import numpy as np
from scipy import stats

import gpflow
from gpflow.test_util import GPflowTestCase

from gpflow import gaussian_utils

class TestLogPdfNormal(GPflowTestCase):
    def test_log_pdf_normal(self):
        rng = np.random.RandomState(123)

        xs = rng.uniform(-1000, 1000, 25)

        normal = stats.norm()
        expected = normal.logpdf(xs)

        with self.test_context() as sess:
            x_ph = tf.placeholder(tf.float64, xs.shape)
            result = gaussian_utils.log_pdf_normal(x_ph)
            result_evald = sess.run(result, feed_dict={x_ph: xs})

        np.testing.assert_array_almost_equal(result_evald, expected, decimal=10)


class TestDerivLogCdfNormal(GPflowTestCase):

    #TODO: lots of copied code in the tests: consider parameterising...?

    def test_deriv_log_cdf_normal(self):
        rng = np.random.RandomState(123)

        xs = rng.uniform(-30, 30, 25)

        normal = stats.norm()
        pdf = normal.pdf(xs)
        expected = np.where(pdf == 0, np.zeros_like(pdf, dtype=np.float64), normal.pdf(xs) / normal.cdf(xs))

        with self.test_context() as sess:
            x_ph = tf.placeholder(tf.float64, xs.shape)
            result = gaussian_utils.deriv_log_cdf_normal(x_ph)
            result_evald = sess.run(result, feed_dict={x_ph: xs})

        np.testing.assert_array_almost_equal(result_evald, expected)

    def test_deriv_log_cdf_normal_inbetween_erf_cody_limit(self):
        rng = np.random.RandomState(123)

        xs = rng.uniform(-gaussian_utils.ERF_CODY_LIMIT1, gaussian_utils.ERF_CODY_LIMIT1, 25)

        normal = stats.norm()
        pdf = normal.pdf(xs)
        expected = np.where(pdf == 0, np.zeros_like(pdf, dtype=np.float64), normal.pdf(xs) / normal.cdf(xs))

        with self.test_context() as sess:
            x_ph = tf.placeholder(tf.float64, xs.shape)
            result = gaussian_utils.deriv_log_cdf_normal(x_ph)
            result_evald = sess.run(result, feed_dict={x_ph: xs})

        np.testing.assert_array_almost_equal(result_evald, expected)

    def test_deriv_log_cdf_normal_lower_than_zero(self):
        rng = np.random.RandomState(123)

        xs = rng.uniform(-25, -gaussian_utils.ERF_CODY_LIMIT1, 25)

        normal = stats.norm()
        pdf = normal.pdf(xs)
        expected = np.where(pdf == 0, np.zeros_like(pdf, dtype=np.float64), normal.pdf(xs) / normal.cdf(xs))

        with self.test_context() as sess:
            x_ph = tf.placeholder(tf.float64, xs.shape)
            result = gaussian_utils.deriv_log_cdf_normal(x_ph)
            result_evald = sess.run(result, feed_dict={x_ph: xs})

        np.testing.assert_array_almost_equal(result_evald, expected)

    def test_deriv_log_cdf_normal_higher(self):
        rng = np.random.RandomState(123)

        xs = rng.uniform(gaussian_utils.ERF_CODY_LIMIT1, 25, 25)

        normal = stats.norm()
        pdf = normal.pdf(xs)
        expected = np.where(pdf == 0, np.zeros_like(pdf, dtype=np.float64), normal.pdf(xs) / normal.cdf(xs))

        with self.test_context() as sess:
            x_ph = tf.placeholder(tf.float64, xs.shape)
            result = gaussian_utils.deriv_log_cdf_normal(x_ph)
            result_evald = sess.run(result, feed_dict={x_ph: xs})

        np.testing.assert_array_almost_equal(result_evald, expected)


