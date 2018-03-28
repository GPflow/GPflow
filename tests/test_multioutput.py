import gpflow
import numpy as np
import pytest

import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk

from gpflow.models import SVGP
from gpflow.kernels import RBF
from gpflow.features import InducingPoints
from gpflow.likelihoods import Gaussian


def predict(model, Xnew, full_cov, full_cov_output):
    m, v = model._build_predict(Xnew, full_cov=full_cov, full_cov_output=full_cov_output)
    m, v = model.enquire_session().run([m, v])
    return m, v


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
    q_mu_1 = np.random.randn(Data.M * Data.P, 1)  # MP x 1
    q_mu_2 = np.reshape(q_mu_1, [Data.M, Data.P])  # M x P
    q_sqrt_2 = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)])  # P x M x M
    q_sqrt_1 = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP

    # Model 1
    kernel_1 = mk.SharedIndependentMok(RBF(Data.D, variance=0.5, lengthscales=1.2), Data.P)
    feature_1 = InducingPoints(Data.X[:Data.M,...].copy())
    m1 = SVGP(Data.X, Data.Y, kernel_1, Gaussian(), feature_1, q_mu=q_mu_1, q_sqrt=q_sqrt_1)
    m1.set_trainable(False)
    m1.q_sqrt.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m1, maxiter=Data.MAXITER)

    # Model 2
    kernel_2 = RBF(Data.D, variance=0.5, lengthscales=1.2)
    feature_2 = InducingPoints(Data.X[:Data.M, ...].copy())
    m2 = SVGP(Data.X, Data.Y, kernel_2, Gaussian(), feature_2, q_mu=q_mu_2, q_sqrt=q_sqrt_2)
    m2.set_trainable(False)
    m2.q_sqrt.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m2, maxiter=Data.MAXITER)

    ## Predict: full_cov = True and full_cov_output = True
    mean1, var1 = predict(m1, Data.Xs, full_cov=True, full_cov_output=True)  # Ns x P, Ns x P x Ns x P
    mean2, var2 = predict(m2, Data.Xs, full_cov=True, full_cov_output=True)  # Ns x P, Ns x P x Ns x P

    np.testing.assert_almost_equal(mean1, mean2, decimal=5)
    np.testing.assert_almost_equal(var1, var2, decimal=4)

    ## Predict: full_cov = True and full_cov_output = False
    mean3, var3 = predict(m1, Data.Xs, full_cov=True, full_cov_output=False)  # Ns x P, P x Ns x Ns
    mean4, var4 = predict(m2, Data.Xs, full_cov=True, full_cov_output=False)  # Ns x P, P x Ns x Ns

    np.testing.assert_almost_equal(mean3, mean4, decimal=5)
    np.testing.assert_almost_equal(mean1, mean3, decimal=5)
    print(np.diagonal(var1, axis1=1, axis2=3).shape)
    np.testing.assert_almost_equal(np.diagonal(var1, axis1=1, axis2=3),
                                   np.transpose(var3, [1, 2, 0]), decimal=4)

    ## Predict: full_cov = False and full_cov_output = True
    mean5, var5 = predict(m1, Data.Xs, full_cov=False, full_cov_output=True)  # Ns x P, Ns x P x P 
    mean6, var6 = predict(m2, Data.Xs, full_cov=False, full_cov_output=True)  # Ns x P, Ns x P x P

    np.testing.assert_almost_equal(mean5, mean6, decimal=5)
    np.testing.assert_almost_equal(mean1, mean5, decimal=5)
    np.testing.assert_almost_equal(var5, var6, decimal=4)
    np.testing.assert_almost_equal(np.diagonal(var1, axis1=0, axis2=2), 
                                   np.transpose(var5, [1, 2, 0]), decimal=4)

    ## Predict: full_cov = False and full_cov_output = False
    mean7, var7 = predict(m1, Data.Xs, full_cov=False, full_cov_output=False)  # Ns x P, Ns x P
    mean8, var8 = predict(m2, Data.Xs, full_cov=False, full_cov_output=False)  # Ns x P, Ns x P

    np.testing.assert_almost_equal(mean7, mean8, decimal=5)
    np.testing.assert_almost_equal(mean1, mean7, decimal=5)
    np.testing.assert_almost_equal(var7, var8, decimal=4)
    np.testing.assert_almost_equal(np.diagonal(np.diagonal(var1, axis1=0, axis2=2)),
                                   var7, decimal=4)