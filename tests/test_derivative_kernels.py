from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

import gpflow

from testing.gpflow_testcase import GPflowTestCase


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


class TestRBFDerivativeKern(GPflowTestCase):
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

        diff_kernel = gpflow.derivative_kernel.RBFDerivativeKern(2, 2)
        diff_kernel.base_kernel.lengthscales = self.lengthscale
        diff_kernel.base_kernel.variance = self.variance

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
        diff_kernel = gpflow.derivative_kernel.RBFDerivativeKern(2, 2)
        diff_kernel.base_kernel.lengthscales = self.lengthscale
        diff_kernel.base_kernel.variance = self.variance

        with self.test_session() as sess:
            with diff_kernel.tf_mode():
                x_free = tf.placeholder('float64')
                diff_kernel.make_tf_array(x_free)
                k = diff_kernel.K(x_ph, x2_ph)
            k_evald = sess.run(k, feed_dict={x_ph:self.x_1, x2_ph:self.x_2 , x_free: diff_kernel.get_free_state()})
        np.testing.assert_array_almost_equal(k_evald, self.expected_kernel2)


    def test_deriv_rbf_kernel_x1_diag(self):
        x_ph = tf.placeholder(tf.float64, [None, 4])

        diff_kernel = gpflow.derivative_kernel.RBFDerivativeKern(2,2)
        diff_kernel.base_kernel.lengthscales = self.lengthscale
        diff_kernel.base_kernel.variance = self.variance

        with self.test_session() as sess:
            with diff_kernel.tf_mode():
                x_free = tf.placeholder('float64')
                diff_kernel.make_tf_array(x_free)
                k = diff_kernel.Kdiag(x_ph)
            k_evald_diag = sess.run(k, feed_dict={x_ph:self.x_data, x_free: diff_kernel.get_free_state()})

        np.testing.assert_array_almost_equal(k_evald_diag, np.diag(self.expected_kernel))


    def test_self_kernel_is_correct(self):
        x1 = np.array([[0.12, 0.], [0.67, 1]])


        x_ph = tf.placeholder(tf.float64, [None, 2])
        kernel = gpflow.derivative_kernel.RBFDerivativeKern(1,1)
        # we leave kernel with default lengthscales and sigma

        expected_raw_kerel = np.array([[ 1., 0.85963276],
                                        [ 0.85963276,  1.]])
        expected_kernel = np.array([[1, -0.4727980199813982], [-0.4727980199813982, 1]])

        with self.test_session() as sess:
            with kernel.tf_mode():
                x_free = tf.placeholder('float64')
                kernel.make_tf_array(x_free)
                k = kernel.K(x_ph)
            k_evald = sess.run(k, feed_dict={x_ph:x1, x_free: kernel.get_free_state()})
            np.testing.assert_array_almost_equal(k_evald, expected_kernel)

    def test_deriv_rbf_kernel_x1_and_x2_different_lengthscales(self):
        # this test is mainly about testing our rbf derivative kernel implementation
        # when the lengthscales vary along the different dimensions.
        # to do this we test the result against the basic derivative kernel
        # where the gradients are calculated via tf.gradients.
        x_ph = tf.placeholder(tf.float64, [None, 4])
        x2_ph = tf.placeholder(tf.float64, [None, 4])
        lengthscales = np.array([1.8, 0.9])
        base_rbf_kern1 = gpflow.kernels.RBF(2, self.variance, lengthscales=lengthscales,
                                            ARD=True)
        base_rbf_kern2 = gpflow.kernels.RBF(2, self.variance, lengthscales=lengthscales,
                                            ARD=True)

        diff_dynamic_kernel = gpflow.derivative_kernel.DifferentialObservationsKernelDynamic(
            2, base_rbf_kern1, 2
        )
        diff_kernel = gpflow.derivative_kernel.RBFDerivativeKern(2, 2, base_kernel=base_rbf_kern2)

        with self.test_session() as sess:
            with diff_kernel.tf_mode():
                x_free = tf.placeholder('float64')
                diff_kernel.make_tf_array(x_free)
                k = diff_kernel.K(x_ph, x2_ph)

            with diff_dynamic_kernel.tf_mode():
                x_free_2 = tf.placeholder('float64')
                diff_dynamic_kernel.make_tf_array(x_free_2)
                k2 = diff_dynamic_kernel.K(x_ph, x2_ph)
            k_evald = sess.run(k, feed_dict={x_ph:self.x_1, x2_ph:self.x_2,
                                             x_free: diff_kernel.get_free_state()})
            k2_evald = sess.run(k2, feed_dict={x_ph:self.x_1, x2_ph:self.x_2,
                                               x_free_2: diff_dynamic_kernel.get_free_state()})
        np.testing.assert_array_almost_equal(k_evald, k2_evald)

    def test_second_derivs(self):

        x1 = np.array([[0.7,0.], [0.21,1], [0.89, 0], [-0.43,2], [0.21, 2], [-0.21, 1]])
        x2 = np.array([[0.32,1], [-0.54,0], [1.54, 2], [0.31,1]])


        x_ph = tf.placeholder(tf.float64, [None, 2])
        x2_ph = tf.placeholder(tf.float64, [None, 2])
        lengthscales = 1.8
        base_rbf_kern1 = gpflow.kernels.RBF(1, self.variance, lengthscales=lengthscales,
                                            ARD=True)
        base_rbf_kern2 = gpflow.kernels.RBF(1, self.variance, lengthscales=lengthscales,
                                            ARD=True)

        diff_dynamic_kernel = gpflow.derivative_kernel.DifferentialObservationsKernelDynamic(
            1, base_rbf_kern1, 1
        )
        diff_kernel = gpflow.derivative_kernel.RBFDerivativeKern(1, 1, base_kernel=base_rbf_kern2)

        with self.test_session() as sess:
            with diff_kernel.tf_mode():
                x_free = tf.placeholder('float64')
                diff_kernel.make_tf_array(x_free)
                k = diff_kernel.K(x_ph, x2_ph)

            with diff_dynamic_kernel.tf_mode():
                x_free_2 = tf.placeholder('float64')
                diff_dynamic_kernel.make_tf_array(x_free_2)
                k2 = diff_dynamic_kernel.K(x_ph, x2_ph)
            k_evald = sess.run(k, feed_dict={x_ph: x1, x2_ph: x2,
                                             x_free: diff_kernel.get_free_state()})
            k2_evald = sess.run(k2, feed_dict={x_ph: x1, x2_ph: x2,
                                               x_free_2: diff_dynamic_kernel.get_free_state()})
        np.testing.assert_array_almost_equal(k_evald, k2_evald)


class TestTensorflowRepeats(GPflowTestCase):
    def test_tesnorflow_repeats(self):
        vector = np.array([0., 4.7, 2.])
        repeats = 3

        expected = np.array([0., 0., 0., 4.7, 4.7, 4.7, 2., 2., 2.])
        x_ph = tf.placeholder(tf.float32, [None])
        repeats_ph = tf.placeholder(tf.int32, [])
        with self.test_session() as sess:
            actual = gpflow.derivative_kernel.tensorflow_repeats(x_ph, repeats_ph)
            actual_evald = sess.run(actual, feed_dict={x_ph:vector, repeats_ph: repeats})
        np.testing.assert_array_almost_equal(expected, actual_evald)


class TestDerivativeFactory(GPflowTestCase):
    def test_picks_up_rbf(self):
        kern = gpflow.kernels.RBF(10)

        deriv_kern = gpflow.derivative_kernel.derivative_kernel_factory(10, 10, kern)

        assert isinstance(deriv_kern,
                          gpflow.derivative_kernel.RBFDerivativeKern)

    def test_falls_back_to_default(self):
        kern = gpflow.kernels.Matern52(10)

        deriv_kern = gpflow.derivative_kernel.derivative_kernel_factory(10, 10, kern)

        assert type(deriv_kern) == gpflow.derivative_kernel.DifferentialObservationsKernelDynamic


