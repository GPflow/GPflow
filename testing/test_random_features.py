
from __future__ import print_function

import numpy as np
import tensorflow as tf

from scipy import linalg as sla

import gpflow
from testing.gpflow_testcase import GPflowTestCase




class TestGPRRandomFeaturesApprox(GPflowTestCase):

    def setUp(self):
        self.kern_cls = gpflow.kernels.RBF
        self.models_to_test = [gpflow.gpr.GPR]

    def test_models_a_gp_well(self):
        """
        Here we're checking that the random features approximation works well at modeling the
        posterior mean and covariance of a GP. As we do not expect it to be exact we are not too
        bothered about them matching up exactly.
        """
        # We define a fucntion we want the GP to be close to
        def func1(x):
            f = np.sin(12 * x) + 0.66 * np.cos(25 * x)
            return f

        # Set up the x and y data
        x_data = np.linspace(0, 10, 40)[:, np.newaxis]
        y_data = func1(x_data)
        x_preds_points = np.linspace(0, 10, 50)[:, np.newaxis]

        # Loop through and test all applicable models for which this should work on.
        for model_cls in self.models_to_test:
            # Sample theta
            model = model_cls(x_data, y_data, self.kern_cls(1))
            model.optimize()
            mean, L_precision = model.linear_weights_posterior()

            # Map new points through to the linear space:
            feature_mapper = create_approx_features_func(model, 1)
            feats_at_sample_points = feature_mapper(x_preds_points)

            # predicted mean and variance. This matches up to Eqn 2.9 of GPML.
            mean_predicted = feats_at_sample_points @ mean
            intermediate = sla.solve_triangular(L_precision, feats_at_sample_points.T, lower=True)
            var_predicted = intermediate.T @ intermediate
            # variance is X * (LL^T)^-1 XT

            # predicted mean and variance from the GP
            mean_from_gp, var_from_gp = model.predict_f_full_cov(x_preds_points)
            var_from_gp = var_from_gp [:, :, 0]

            # And now we test whether these are correct. To create some reasonable thresholds
            # we want to the approximations to be within we look at the maximum absolute value
            # of the true function.
            mean_abs_diffs = np.abs(mean_from_gp - mean_predicted)
            mean_thresh = np.max(np.abs(mean_from_gp)) * 0.05
            var_abs_diffs = np.abs(var_predicted - var_from_gp)
            var_thresh = np.max(np.abs(var_from_gp)) * 0.05

            np.testing.assert_array_less(mean_abs_diffs, mean_thresh)
            np.testing.assert_array_less(var_abs_diffs, var_thresh)

    def test_samples_find_maximum_of_quadratic(self):
        """
        The random features are useful in particular for Thompson sampling. Here we want to check
        that when sampling from the random features you get sensible samples. To test this we make
        a very obvious quadratic function. We ensure that the samples maximum is close to this
        point.
        """

        # Function is a simple quadratic with a very clear maximum
        maximum = 4.7
        def func(X):
            return -(X -maximum)**2


        # Set up the x and y data
        x_data = np.linspace(0, 10, 40)[:, np.newaxis]
        y_data = func(x_data)
        x_preds_points = np.linspace(0, 10, 50)[:, np.newaxis]
        rng = np.random.RandomState(100)

        # Loop through and test all applicable models for which this should work on.
        for model_cls in self.models_to_test:
            # Sample theta
            model = model_cls(x_data, y_data, self.kern_cls(1))
            model.optimize()
            mean, L_precision = model.linear_weights_posterior()
            sampled_var = sla.solve_triangular(L_precision.T, rng.randn(mean.shape[0]),
                                               lower=False)[:, None]
            theta_sample = mean + sampled_var

            # Map new points through to the linear space:
            feature_mapper = create_approx_features_func(model, 1)
            feats_at_sample_points = feature_mapper(x_preds_points)

            # Compute the function sample:
            predicted_locs = feats_at_sample_points @ theta_sample

            # Make sure the function sample maximum is close to where
            maximum_predicted_loc = np.argmax(predicted_locs)
            x_at_max = x_preds_points[maximum_predicted_loc]
            np.testing.assert_array_less(np.abs(x_at_max - maximum), 0.1)





def create_approx_features_func(model, data_dim):
    """
    Creates a function that maps features to there linear approximations
    """
    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)

    with tf_graph.as_default():
        x_ph = tf.placeholder(tf.float64, [None, data_dim])

        with model.kern.tf_mode():
            x_free = tf.placeholder('float64')
            model.kern.make_tf_array(x_free)
            feats_func = model.kern.create_feature_map_func(model.random_seed_for_random_features)
            feats = feats_func(x_ph)

    def approx_features_func(X):
        feats_evald = tf_sess.run(feats, feed_dict={x_ph:X, x_free: model.kern.get_free_state()})
        return feats_evald

    return approx_features_func


