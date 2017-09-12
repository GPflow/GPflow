from __future__ import absolute_import, print_function
import gpflow
import tensorflow as tf
import numpy as np
import unittest

from testing.gpflow_testcase import GPflowTestCase
from .reference import referenceRbfKernel, referenceArcCosineKernel, referencePeriodicKernel



class TestDifferentialObservationsKernelDynamic(GPflowTestCase):

    def setUp(self):
        self.x_data = np.array([[[1.7, 0], [3.2, 0]], [[0.32, 0], [1.4, 1]], [[3.1, 1], [2.1,1]]])
        self.x_data = np.concatenate((self.x_data[:,:,0], self.x_data[:,:,1]), axis=1)
        self.lengthscale = 1.8
        self.variance = 2.32
        self.expected_kernel = np.array([
            [2.32, 0.5826851055555556, -0.20867138484987047],
            [0.5826851055555556, self.variance/(self.lengthscale**2 ),-0.14669897751467237],
            [-0.20867138484987047 , -0.14669897751467237, self.variance/(self.lengthscale**4)]
        ])
        # ^ calculated by hand

        self.x_1 = np.array([[[0.1, 1], [-0.21, 0]], [[0.32, 0], [-0.67, 1]], [[0.3,2], [-0.7, 0]]])
        self.x_1 = np.concatenate((self.x_1[:,:,0], self.x_1[:,:,1]), axis=1)
        self.x_2 = np.array([[[0.8, 0], [-0.97, 0]], [[0.11, 0], [-0.89, 1]], [[-0.5, 0], [0.1, 0]]])
        self.x_2 = np.concatenate((self.x_2[:,:,0], self.x_2[:,:,1]), axis=1)
        self.expected_kernel2 = np.array([
            [0.42509858395061728,  0.0013992965973174815, -0.40042949444444437],
            [-0.20445172499999995, 0.69535566997561338, 0.45355748709876537],
            [-0.62867783977290037, -0.041062615565529925, -0.47161189376619417]
        ])
        # ^ calculated by evaluating the derivs by hand.

    def test_deriv_rbf_kernel_only_x(self):
        x_ph = tf.placeholder(tf.float64, [None, 4])

        rbf_kernel = gpflow.kernels.RBF(2, variance=self.variance, lengthscales=self.lengthscale)
        diff_kernel = gpflow.derivative_kernel.DifferentialObservationsKernelDynamic(
            2, rbf_kernel, 2
        )

        with self.test_session() as sess:
            with diff_kernel.tf_mode():
                x_free = tf.placeholder('float64')
                diff_kernel.make_tf_array(x_free)
                k = diff_kernel.K(x_ph)
            k_evald = sess.run(k, feed_dict={x_ph:self.x_data, x_free: diff_kernel.get_free_state()})

        np.testing.assert_array_almost_equal(k_evald, self.expected_kernel)


    def test_deriv_rbf_kernel_x1_and_x2(self):


        x_ph = tf.placeholder(tf.float64, [None, 4])
        x2_ph = tf.placeholder(tf.float64, [None, 4])

        rbf_kernel = gpflow.kernels.RBF(2, variance=self.variance, lengthscales=self.lengthscale)
        diff_kernel = gpflow.derivative_kernel.DifferentialObservationsKernelDynamic(
            2, rbf_kernel, 2
        )

        with self.test_session() as sess:
            with diff_kernel.tf_mode():
                x_free = tf.placeholder('float64')
                diff_kernel.make_tf_array(x_free)
                k_base = rbf_kernel.K(x_ph, x2_ph)
                k = diff_kernel.K(x_ph, x2_ph)
            k_evald = sess.run(k, feed_dict={x_ph:self.x_1, x2_ph:self.x_2 , x_free: diff_kernel.get_free_state()})
            print(k_evald)
            #
            # k_base_evald = sess.run(k_base, feed_dict={x_ph:self.x_1, x2_ph:self.x_2 , x_free: diff_kernel.get_free_state()})
            # print(k_base_evald)
        np.testing.assert_array_almost_equal(k_evald, self.expected_kernel2)


    def test_deriv_rbf_kernel_x1_diag(self):
        x_ph = tf.placeholder(tf.float64, [None, 4])

        rbf_kernel = gpflow.kernels.RBF(2, variance=self.variance, lengthscales=self.lengthscale)
        diff_kernel = gpflow.derivative_kernel.DifferentialObservationsKernelDynamic(
            2, rbf_kernel, 2
        )

        with self.test_session() as sess:
            with diff_kernel.tf_mode():
                x_free = tf.placeholder('float64')
                diff_kernel.make_tf_array(x_free)
                k = diff_kernel.Kdiag(x_ph)
            k_evald_diag = sess.run(k, feed_dict={x_ph:self.x_data, x_free: diff_kernel.get_free_state()})

        np.testing.assert_array_almost_equal(k_evald_diag, np.diag(self.expected_kernel))

