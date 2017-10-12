from __future__ import print_function

import unittest
import functools

import numpy as np
from scipy import linalg as sla

import gpflow
from gpflow.test_util import GPflowTestCase


class TestRandomFeaturesApprox(GPflowTestCase):
    def setUp(self):
        self.kern_cls = gpflow.kernels.RBF
        self.rng = np.random.RandomState(1021)

        def create_gpr(x_data, y_data, kern):
            return gpflow.models.GPR(x_data, y_data, kern)

        def create_svgp(whiten, x_data, y_data, kern):
            num_inducing = 100
            indices = self.rng.choice(x_data.shape[0], num_inducing, replace=False)
            initial_z = np.copy(x_data[indices, :])
            return gpflow.models.SVGP(x_data, y_data, kern, gpflow.likelihoods.Gaussian(),
                                    Z=initial_z, whiten=whiten)

        self.models_to_test = {"svgp_whitened": functools.partial(create_svgp, True),
                               "svgp_nonwhitened": functools.partial(create_svgp, False),
                               "gpr": create_gpr}

    def test_models_a_gp_well(self):
        """
        Here we're checking that the random features approximation works okay at modeling the
        posterior mean and covariance of a GP. As we do not expect it to be exact we are not too
        bothered about them matching up exactly, and so some of the thresholds are quite rough.
        Plotting code can be used by human below to check that sensible.
        """
        # We define a fucntion we want the GP to be close to
        def func1(x):
            f = x ** 3 - 2 * x ** 2 + x
            return f

        # Set up the x and y data
        x_data = np.linspace(0, 1, 100)[:, np.newaxis]
        y_data = func1(x_data)
        x_preds_points = np.linspace(0.2, 0.8, 50)[:, np.newaxis]

        # Loop through and test all applicable models for which this should work on.
        for name, model_cls in self.models_to_test.items():
            # Sample theta
            model = model_cls(x_data, y_data, self.kern_cls(1, lengthscales=0.5,
                                                            num_features_to_approx=5000))
            model.likelihood.variance = 1e-6
            model.likelihood.variance.set_trainable(False)
            model.compile()
            optimizer = gpflow.train.ScipyOptimizer(options=dict(maxiter=100))
            optimizer.minimize(model)
            mean, prec_or_var, var_flag = model.linear_weights_posterior()
            L = sla.cholesky(prec_or_var, lower=True)

            # Map new points through to the linear space:
            feats_at_sample_points = model.kern.feature_map(x_preds_points)

            # predicted mean and variance. This matches up to Eqn 2.9 of GPML.
            mean_predicted = feats_at_sample_points @ mean
            if var_flag:
                # using the Cholesky of the variance
                intermediate = L.T @ feats_at_sample_points.T
            else:
                # using the Cholesky of the precision.
                intermediate = sla.solve_triangular(L, feats_at_sample_points.T, lower=True)
            var_predicted = intermediate.T @ intermediate
            # variance is X * (LL^T)^-1 XT

            # predicted mean and variance from the GP
            mean_from_gp, var_from_gp = model.predict_f_full_cov(x_preds_points)
            var_from_gp = var_from_gp [:, :, 0]

            # And now we test whether these are correct. To create some reasonable thresholds
            # we want to the approximations to be within we look at the maximum absolute value
            # of the true function.
            mean_abs_diffs = np.abs(mean_from_gp - mean_predicted)
            mean_thresh = np.max(np.abs(mean_from_gp)) * 0.1
            var_abs_diffs = np.abs(var_predicted - var_from_gp)
            var_thresh = np.max(np.abs(var_from_gp)) * 0.2

            # The plotting
            PLOT = False
            if PLOT:
                import matplotlib.pyplot as plt
                plt.plot(x_preds_points, mean_from_gp, '^', label="From_gp")
                plt.plot(x_preds_points, mean_predicted, 's', label="From_random_feats")
                plt.plot(x_data, y_data, 'o', label="training data")
                plt.legend()
                plt.title(name)
                plt.show()

            self.assertTrue(np.mean(np.less(mean_abs_diffs, mean_thresh), dtype=np.float32) > 0.8,
                                         msg="Mean match failed for {}".format(name))
            self.assertTrue(np.mean(np.less(var_abs_diffs, var_thresh), dtype=np.float32) > 0.8,
                                         msg="Var match failed for {}".format(name))

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
        x_data = np.linspace(0, 10, 100)[:, np.newaxis]
        y_data = func(x_data)
        x_preds_points = np.linspace(0, 10, 200)[:, np.newaxis]
        rng = np.random.RandomState(100)

        # Loop through and test all applicable models for which this should work on.
        for name, model_cls in self.models_to_test.items():
            # Sample theta
            model = model_cls(x_data, y_data, self.kern_cls(1))
            # We set the likelihood variance low and untrainable as otherwise we found that
            # sometimes the SVGP model would not train very well and would try to explain
            # everything by noise.
            model.likelihood.variance = 1e-6
            model.likelihood.variance.set_trainable(False)
            model.compile()
            optimizer = gpflow.train.ScipyOptimizer(options=dict(maxiter=100))
            optimizer.minimize(model)
            mean, prec_or_var, var_flag = model.linear_weights_posterior()

            L = sla.cholesky(prec_or_var + 1e-6 * np.eye(prec_or_var.shape[0]), lower=True)
            if var_flag:
                # got given a variance matrix so can use the Cholesky as is to get samples.
                sampled_var = (L @ rng.randn(mean.shape[0]))[:, None]
            else:
                # got given a Precision matrix so need to invert the transpose to get samples.
                sampled_var = sla.solve_triangular(L.T, rng.randn(mean.shape[0]),
                                               lower=False)[:, None]
            theta_sample = mean + sampled_var

            # Map new points through to the linear space:
            feats_at_sample_points = model.kern.feature_map(x_preds_points)

            # Compute the function sample:
            predicted_locs = feats_at_sample_points @ theta_sample

            # Make sure the function sample maximum is close to where
            maximum_predicted_loc = np.argmax(predicted_locs)
            x_at_max = x_preds_points[maximum_predicted_loc]
            np.testing.assert_array_less(np.abs(x_at_max - maximum), 0.25,
                                         err_msg="Failed for model class: {}".format(name))


if __name__ == "__main__":
    unittest.main()





