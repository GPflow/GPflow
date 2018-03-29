import gpflow
import numpy as np
import pytest

import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk

from gpflow.models import SVGP
from gpflow.kernels import RBF
from gpflow.features import InducingPoints
from gpflow.likelihoods import Gaussian
from gpflow.test_util import session_context


def predict(sess, model, Xnew, full_cov, full_cov_output):
    m, v = model._build_predict(Xnew, full_cov=full_cov, full_cov_output=full_cov_output)
    return sess.run([m, v])


def predict_all(sess, models, Xnew, full_cov, full_cov_output):
    ms, vs = [], []
    for model in models:
        m, v = predict(sess, model, Xnew, full_cov, full_cov_output)
        ms.append(m)
        vs.append(v)
    return ms, vs 

def assert_all_array_elements_almost_equal(arr, decimal):
    for i in range(len(arr) - 1):
        np.testing.assert_almost_equal(arr[i], arr[i+1], decimal=decimal)


class Data:
    X = np.random.rand(100)[:, None] * 10 - 5
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))
    Ptrue = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])
    Y = np.matmul(G, Ptrue)
    Y += np.random.randn(*Y.shape) * [0.2, 0.2, 0.2]
    Xs = np.linspace(-6, 6, 5)[:, None]

    D = 1
    M = 3
    L = 2
    P = 3
    MAXITER = int(15e2)

def test_shared_independent_mok_with_inducing_points():
    np.random.seed(0)

    with session_context() as sess:

        # Model 1
        q_mu_1 = np.random.randn(Data.M * Data.P, 1)  # MP x 1
        q_sqrt_1 = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP
        kernel_1 = mk.SharedIndependentMok(RBF(Data.D, variance=0.5, lengthscales=1.2), Data.P)
        feature_1 = InducingPoints(Data.X[:Data.M,...].copy())
        m1 = SVGP(Data.X, Data.Y, kernel_1, Gaussian(), feature_1, q_mu=q_mu_1, q_sqrt=q_sqrt_1)
        m1.set_trainable(False)
        m1.q_sqrt.set_trainable(True)
        gpflow.training.ScipyOptimizer().minimize(m1, maxiter=Data.MAXITER)

        # Model 2
        q_mu_2 = np.reshape(q_mu_1, [Data.M, Data.P])  # M x P
        q_sqrt_2 = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)])  # P x M x M
        kernel_2 = RBF(Data.D, variance=0.5, lengthscales=1.2)
        feature_2 = InducingPoints(Data.X[:Data.M, ...].copy())
        m2 = SVGP(Data.X, Data.Y, kernel_2, Gaussian(), feature_2, q_mu=q_mu_2, q_sqrt=q_sqrt_2)
        m2.set_trainable(False)
        m2.q_sqrt.set_trainable(True)
        gpflow.training.ScipyOptimizer().minimize(m2, maxiter=Data.MAXITER)

        # Model 3
        q_mu_3 = np.reshape(q_mu_1, [Data.M, Data.P])  # M x P
        q_sqrt_3 = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)])  # P x M x M
        kernel_3 = mk.SharedIndependentMok(RBF(Data.D, variance=0.5, lengthscales=1.2), Data.P)
        feature_3 = mf.SharedIndependentMof(InducingPoints(Data.X[:Data.M, ...].copy()))
        m3 = SVGP(Data.X, Data.Y, kernel_3, Gaussian(), feature_3, q_mu=q_mu_3, q_sqrt=q_sqrt_3)
        m3.set_trainable(False)
        m3.q_sqrt.set_trainable(True)
        gpflow.training.ScipyOptimizer().minimize(m3, maxiter=Data.MAXITER)

        # Check equality of log likelihood
        np.testing.assert_allclose(m1.compute_log_likelihood(), m2.compute_log_likelihood())
        np.testing.assert_allclose(m3.compute_log_likelihood(), m2.compute_log_likelihood())

        # Predict: full_cov = True and full_cov_output = True
        means_tt, vars_tt = predict_all(sess, [m1, m2, m3], Data.Xs, full_cov=True, full_cov_output=True)
        # Predict: full_cov = True and full_cov_output = False
        means_tf, vars_tf = predict_all(sess, [m1, m2, m3], Data.Xs, full_cov=True, full_cov_output=False)
        # Predict: full_cov = False and full_cov_output = True
        means_ft, vars_ft = predict_all(sess, [m1, m2, m3], Data.Xs, full_cov=False, full_cov_output=True)
        # Predict: full_cov = False and full_cov_output = False
        means_ff, vars_ff = predict_all(sess, [m1, m2, m3], Data.Xs, full_cov=False, full_cov_output=False)

        # check equality of all the means
        all_means = means_tt + means_tf + means_ft + means_ff
        assert_all_array_elements_almost_equal(all_means, decimal=5)

        # check equality of all the variances within a category 
        # (e.g. full_cov=True and full_cov_output=False)
        all_vars = [vars_tt, vars_tf, vars_ft, vars_ff]
        _ = [assert_all_array_elements_almost_equal(var, decimal=4) for var in all_vars]

        # Here we check that the variance in different categories are equal
        # after transforming to the right shape. 
        var_tt = vars_tt[0]  # N x P x N x P
        var_tf = vars_tf[0]  # P x N x N
        var_ft = vars_ft[0]  # N x P x P
        var_ff = vars_ff[0]  # N x P
        
        np.testing.assert_almost_equal(np.diagonal(var_tt, axis1=1, axis2=3),
                                       np.transpose(var_tf, [1, 2, 0]), decimal=4)
        np.testing.assert_almost_equal(np.diagonal(var_tt, axis1=0, axis2=2), 
                                       np.transpose(var_ft, [1, 2, 0]), decimal=4)
        np.testing.assert_almost_equal(np.diagonal(np.diagonal(var_tt, axis1=0, axis2=2)),
                                       var_ff, decimal=4)