import gpflow
import numpy as np
import pytest

import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk

from gpflow.models import SVGP
from gpflow.kernels import RBF
from gpflow.features import InducingPoints
from gpflow.likelihoods import Gaussian
from gpflow.test_util import session_tf

float_type = gpflow.settings.float_type
np.random.seed(1)

def predict(sess, model, Xnew, full_cov, full_output_cov):
    m, v = model._build_predict(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
    return sess.run([m, v])


def predict_all(sess, models, Xnew, full_cov, full_output_cov):
    ms, vs = [], []
    for model in models:
        m, v = predict(sess, model, Xnew, full_cov, full_output_cov)
        ms.append(m)
        vs.append(v)
    return ms, vs


def assert_all_array_elements_almost_equal(arr, decimal):
    for i in range(len(arr) - 1):
        np.testing.assert_almost_equal(arr[i], arr[i+1], decimal=decimal)


def check_equality_predictions(sess, models, decimal=4):
    log_likelihoods = [m.compute_log_likelihood() for m in models]

    # Check equality of log likelihood
    # assert_all_array_elements_almost_equal(log_likelihoods, decimal=5)

    # Predict: full_cov = True and full_output_cov = True
    means_tt, vars_tt = predict_all(sess, models, Data.Xs, full_cov=True, full_output_cov=True)
    # Predict: full_cov = True and full_output_cov = False
    means_tf, vars_tf = predict_all(sess, models, Data.Xs, full_cov=True, full_output_cov=False)
    # Predict: full_cov = False and full_output_cov = True
    means_ft, vars_ft = predict_all(sess, models, Data.Xs, full_cov=False, full_output_cov=True)
    # Predict: full_cov = False and full_output_cov = False
    means_ff, vars_ff = predict_all(sess, models, Data.Xs, full_cov=False, full_output_cov=False)

    # check equality of all the means
    all_means = means_tt + means_tf + means_ft + means_ff
    assert_all_array_elements_almost_equal(all_means, decimal=decimal)

    # check equality of all the variances within a category
    # (e.g. full_cov=True and full_output_cov=False)
    all_vars = [vars_tt, vars_tf, vars_ft, vars_ff]
    _ = [assert_all_array_elements_almost_equal(var, decimal=decimal) for var in all_vars]

    # Here we check that the variance in different categories are equal
    # after transforming to the right shape.
    var_tt = vars_tt[0]  # N x P x N x P
    var_tf = vars_tf[0]  # P x N x N
    var_ft = vars_ft[0]  # N x P x P
    var_ff = vars_ff[0]  # N x P

    np.testing.assert_almost_equal(np.diagonal(var_tt, axis1=1, axis2=3),
                                   np.transpose(var_tf, [1, 2, 0]), decimal=decimal)
    np.testing.assert_almost_equal(np.diagonal(var_tt, axis1=0, axis2=2),
                                   np.transpose(var_ft, [1, 2, 0]), decimal=decimal)
    np.testing.assert_almost_equal(np.diagonal(np.diagonal(var_tt, axis1=0, axis2=2)),
                                   var_ff, decimal=decimal)


def make_sqrt_data(rng, N, M):
    return np.array([np.tril(rng.randn(M, M)) for _ in range(N)])  # N x M x M


def expand_cov(G, W):
    ''' G is L x M x M
        W is L x L
        Output is LM x LM
    '''
    L, M, _ = G.shape
    O = np.zeros((L * M, L * M))
    for l1 in range(L):
        for l2 in range(L):
            O[l1 * M:(l1 + 1) * M, l2 * M:(l2 + 1) * M] = W[l1, l2] * G[l1, :, :]
    return O[None, :, :]


def q_sqrts_to_Q_sqrt(q_sqrt, W):
    ''' G is L x M x M
        W is L x L
        Output is LM x LM
    '''
    cov = np.matmul(q_sqrt, q_sqrt.transpose(0, 2, 1))
    Cov = expand_cov(cov, W)
    return np.linalg.cholesky(Cov)


def mus_to_Mu(mu, W):
    M, L = mu.shape
    Mu = np.zeros((M * L, 1))
    for l1 in range(L):
        for l2 in range(L):
            Mu[l1 * M:(l1 + 1) * M, 0] += mu[:, l2] * W[l1, l2]
    return Mu


class Data:
    N = 20
    Ntest = 5
    D = 1  # input dimension
    M = 3  # inducing points
    L = 2  # latent gps
    P = 3  # output dimension
    MAXITER = int(15e2)

    X = np.random.rand(N)[:, None] * 10 - 5
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))
    Ptrue = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
    Y = np.matmul(G, Ptrue)
    Y += np.random.randn(*Y.shape) * [0.2, 0.2, 0.2]
    Xs = np.linspace(-6, 6, Ntest)[:, None]


class DataMixedKernelWithEye(Data):
    M = 7
    L = 3
    W = np.eye(L)
    G = np.hstack([0.5 * np.sin(3 * Data.X) + Data.X, 
                   Data.X,
                   3.0 * np.cos(Data.X) - Data.X])  # N x P

    rng = np.random.RandomState(0)
    mu_data = rng.randn(M, L)  # M x L
    sqrt_data = make_sqrt_data(rng, L, M)  # L x M x M

    mu_data_full = mus_to_Mu(mu_data, W)  # 1 x ML
    sqrt_data_full = q_sqrts_to_Q_sqrt(sqrt_data, W)  # 1 x LM x LM

    Y = np.matmul(G, W)
    Y += np.random.randn(*Y.shape) * np.ones((L,)) * 0.2



def test_shared_independent_mok(session_tf):
    """
    In this test we use the same kernel and the same inducing features
    for each of the outputs. The outputs are considered to be uncorrelated.
    This is how GPflow handled multiple outputs before the multioutput framework was added.
    We compare three models here:
        1) an inefffient one, where we use a SharedIndepedentMok with InducingPoints.
           This combination will uses a Kff of size N x P x N x P, Kfu if size N x P x M x P
           which is extremely inefficient as most of the elements are zero.
        2) efficient: SharedIndependentMok and SharedIndependentMof
           This combinations uses the most efficient form of matrices
        3) the old way, efficient way: using Kernel and InducingPoints
        Model 2) and 3) follow more or less the same code path.
    """
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

    check_equality_predictions(session_tf, [m1, m2, m3])



def test_seperate_independent_mok(session_tf):
    """
    We use different independent kernels for each of the output dimensions.
    We can achieve this in two ways:
        1) efficient: SeparateIndependentMok with Shared/SeparateIndependentMof
        2) inefficient: SeparateIndependentMok with InducingPoints
    However, both methods should return the same conditional,
    and after optimization return the same log likelihood.
    """
    # Model 1 (INefficient)
    q_mu_1 = np.random.randn(Data.M * Data.P, 1)
    q_sqrt_1 = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP
    kern_list_1 = [RBF(Data.D, variance=0.5, lengthscales=1.2) for _ in range(Data.P)]
    kernel_1 = mk.SeparateIndependentMok(kern_list_1)
    feature_1 = InducingPoints(Data.X[:Data.M,...].copy())
    m1 = SVGP(Data.X, Data.Y, kernel_1, Gaussian(), feature_1, q_mu=q_mu_1, q_sqrt=q_sqrt_1)
    m1.set_trainable(False)
    m1.q_sqrt.set_trainable(True)
    m1.q_mu.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m1, maxiter=Data.MAXITER)

    # Model 2 (efficient)
    q_mu_2 = np.random.randn(Data.M, Data.P)
    q_sqrt_2 = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)])  # P x M x M
    kern_list_2 = [RBF(Data.D, variance=0.5, lengthscales=1.2) for _ in range(Data.P)]
    kernel_2 = mk.SeparateIndependentMok(kern_list_2)
    feature_2 = mf.SharedIndependentMof(InducingPoints(Data.X[:Data.M, ...].copy()))
    m2 = SVGP(Data.X, Data.Y, kernel_2, Gaussian(), feature_2, q_mu=q_mu_2, q_sqrt=q_sqrt_2)
    m2.set_trainable(False)
    m2.q_sqrt.set_trainable(True)
    m2.q_mu.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m2, maxiter=Data.MAXITER)

    check_equality_predictions(session_tf, [m1, m2])


def test_seperate_independent_mof(session_tf):
    """
    Same test as above but we use different (i.e. separate) inducing features
    for each of the output dimensions.
    """
    np.random.seed(0)

    # Model 1 (INefficient)
    q_mu_1 = np.random.randn(Data.M * Data.P, 1)
    q_sqrt_1 = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP
    kernel_1 = mk.SharedIndependentMok(RBF(Data.D, variance=0.5, lengthscales=1.2), Data.P)
    feature_1 = InducingPoints(Data.X[:Data.M,...].copy())
    m1 = SVGP(Data.X, Data.Y, kernel_1, Gaussian(), feature_1, q_mu=q_mu_1, q_sqrt=q_sqrt_1)
    m1.set_trainable(False)
    m1.q_sqrt.set_trainable(True)
    m1.q_mu.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m1, maxiter=Data.MAXITER)

    # Model 2 (efficient)
    q_mu_2 = np.random.randn(Data.M, Data.P)
    q_sqrt_2 = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)])  # P x M x M
    kernel_2 = mk.SharedIndependentMok(RBF(Data.D, variance=0.5, lengthscales=1.2), Data.P)
    feat_list_2 = [InducingPoints(Data.X[:Data.M, ...].copy()) for _ in range(Data.P)]
    feature_2 = mf.SeparateIndependentMof(feat_list_2)
    m2 = SVGP(Data.X, Data.Y, kernel_2, Gaussian(), feature_2, q_mu=q_mu_2, q_sqrt=q_sqrt_2)
    m2.set_trainable(False)
    m2.q_sqrt.set_trainable(True)
    m2.q_mu.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m2, maxiter=Data.MAXITER)

    check_equality_predictions(session_tf, [m1, m2])


def test_mixed_mok_with_Id_vs_independent_mok(session_tf):
    # Independent model
    k1 = mk.SharedIndependentMok(RBF(DataMixedKernelWithEye.D, variance=0.5, lengthscales=1.2), DataMixedKernelWithEye.L)
    f1 = InducingPoints(DataMixedKernelWithEye.X[:DataMixedKernelWithEye.M, ...].copy())
    m1 = SVGP(DataMixedKernelWithEye.X, DataMixedKernelWithEye.Y, k1, Gaussian(), f1,
              q_mu=DataMixedKernelWithEye.mu_data_full, q_sqrt=DataMixedKernelWithEye.sqrt_data_full)
    m1.set_trainable(False)
    m1.q_sqrt.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m1, maxiter=DataMixedKernelWithEye.MAXITER)

    # Mixed Model
    kern_list = [RBF(DataMixedKernelWithEye.D, variance=0.5, lengthscales=1.2) for _ in range(DataMixedKernelWithEye.L)]
    k2 = mk.SeparateMixedMok(kern_list, DataMixedKernelWithEye.W)
    f2 = InducingPoints(DataMixedKernelWithEye.X[:DataMixedKernelWithEye.M, ...].copy())
    m2 = SVGP(DataMixedKernelWithEye.X, DataMixedKernelWithEye.Y, k2, Gaussian(), f2,
              q_mu=DataMixedKernelWithEye.mu_data_full, q_sqrt=DataMixedKernelWithEye.sqrt_data_full)
    m2.set_trainable(False)
    m2.q_sqrt.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m2, maxiter=DataMixedKernelWithEye.MAXITER)

    check_equality_predictions(session_tf, [m1, m2])