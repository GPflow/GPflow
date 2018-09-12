import gpflow
import numpy as np
import pytest
import scipy
import tensorflow as tf

import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk

from gpflow.features import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP
from gpflow.test_util import session_tf
from gpflow.training import ScipyOptimizer
from gpflow.conditionals import _sample_mvn, sample_conditional

float_type = gpflow.settings.float_type
np.random.seed(1)

# ------------------------------------------
# Helpers
# ------------------------------------------

def predict(sess, model, Xnew, full_cov, full_output_cov):
    m, v = model._build_predict(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
    return sess.run([m, v])


def predict_all(sess, models, Xnew, full_cov, full_output_cov):
    """
    Returns the mean and variance of f(Xnew) for each model in `models`.
    """
    ms, vs = [], []
    for model in models:
        m, v = predict(sess, model, Xnew, full_cov, full_output_cov)
        ms.append(m)
        vs.append(v)
    return ms, vs


def assert_all_array_elements_almost_equal(arr, decimal):
    """
    Check if consecutive elements of `arr` are almost equal.
    """
    for i in range(len(arr) - 1):
        np.testing.assert_almost_equal(arr[i], arr[i+1], decimal=decimal)


def check_equality_predictions(sess, models, decimal=4):
    """
    Executes a couple of checks to compare the equality of predictions
    of different models. The models should be configured with the same
    training data (X, Y). The following checks are done:
    - check if log_likelihood is (almost) equal for all models
    - check if predicted mean is (almost) equal
    - check if predicted variance is (almost) equal.
      All possible variances over the inputs and outputs are calculated
      and equality is checked.
    - check if variances within model are consistent. Parts of the covariance
      matrices should overlap, and this is tested.
    """

    log_likelihoods = [m.compute_log_likelihood() for m in models]

    # Check equality of log likelihood
    assert_all_array_elements_almost_equal(log_likelihoods, decimal=5)

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
    var_tf = vars_tf[0]  # P x N x c
    var_ft = vars_ft[0]  # N x P x P
    var_ff = vars_ff[0]  # N x P

    np.testing.assert_almost_equal(np.diagonal(var_tt, axis1=1, axis2=3),
                                   np.transpose(var_tf, [1, 2, 0]), decimal=decimal)
    np.testing.assert_almost_equal(np.diagonal(var_tt, axis1=0, axis2=2),
                                   np.transpose(var_ft, [1, 2, 0]), decimal=decimal)
    np.testing.assert_almost_equal(np.diagonal(np.diagonal(var_tt, axis1=0, axis2=2)),
                                   var_ff, decimal=decimal)


def expand_cov(q_sqrt, W):
    """
    :param G: cholesky of covariance matrices, L x M x M
    :param W: mixing matrix (square),  L x L
    :return: cholesky of 1 x LM x LM covariance matrix
    """
    q_cov = np.matmul(q_sqrt, q_sqrt.transpose(0, 2, 1))  # L x M x M
    q_cov_expanded = scipy.linalg.block_diag(*q_cov)  # LM x LM
    q_sqrt_expanded = np.linalg.cholesky(q_cov_expanded)  # LM x LM
    return q_sqrt_expanded[None, ...]


def create_q_sqrt(M, L):
    """ returns an array of L lower triangular matrices of size M x M """
    return np.array([np.tril(np.random.randn(M, M)) for _ in range(L)])  # L x M x M


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------

class Data:
    N, Ntest = 20, 5
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
    """ Note in this class L == P """
    M, L = 4, 3
    W = np.eye(L)

    G = np.hstack([0.5 * np.sin(3 * Data.X) + Data.X,
                   3.0 * np.cos(Data.X) - Data.X,
                   1.0 + Data.X])  # N x P

    mu_data = np.random.randn(M, L)  # M x L
    sqrt_data = create_q_sqrt(M, L)  # L x M x M

    mu_data_full = (mu_data @ W).reshape(-1, 1)  # ML x 1
    sqrt_data_full = expand_cov(sqrt_data, W)  # 1 x LM x LM

    Y = np.matmul(G, W)
    Y += np.random.randn(*Y.shape) * np.ones((L,)) * 0.2


class DataMixedKernel(Data):
    M = 5
    L = 2
    P = 3
    W = np.random.randn(P, L)
    G = np.hstack([0.5 * np.sin(3 * Data.X) + Data.X,
                   3.0 * np.cos(Data.X) - Data.X])  # N x L

    mu_data = np.random.randn(M, L)  # M x L
    sqrt_data = create_q_sqrt(M, L)  # L x M x M

    Y = np.matmul(G, W.T)
    Y += np.random.randn(*Y.shape) * np.ones((P,)) * 0.1

# ------------------------------------------
# Test sample conditional
# ------------------------------------------


@pytest.mark.parametrize("cov_structure", ["full", "diag"])
@pytest.mark.parametrize("num_samples", [None, 1, 10])
def test_sample_mvn(session_tf, cov_structure, num_samples):
    """
    Draws 10,000 samples from a distribution
    with known mean and covariance. The test checks
    if the mean and covariance of the samples is
    close to the true mean and covariance.
    """

    N, D = 10000, 2
    means = tf.ones((N, D), dtype=float_type)
    if cov_structure == "full":
        covs = tf.eye(D, batch_shape=[N], dtype=float_type)
    elif cov_structure == "diag":
        covs = tf.ones((N, D), dtype=float_type)

    samples = _sample_mvn(means, covs, cov_structure, num_samples=num_samples)
    value = session_tf.run(samples)

    if num_samples is None:
        assert value.shape == (N, D)
    else:
        assert value.shape == (num_samples, N, D)
        value = value.reshape(-1, D)

    samples_mean = np.mean(value, axis=0)
    samples_cov = np.cov(value, rowvar=False)
    np.testing.assert_array_almost_equal(samples_mean, [1., 1.], decimal=1)
    np.testing.assert_array_almost_equal(samples_cov, [[1., 0.], [0., 1.]], decimal=1)


def _create_placeholder_dict(values):
    return {name: tf.placeholder(float_type, shape=arr.shape) for name, arr in values.items()}

def _create_feed_dict(placeholders_dict, value_dict):
    return {placeholder: value_dict[name] for name, placeholder in placeholders_dict.items()}


@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("full_cov,full_output_cov", [(False, False), (False, True), (True, False)])
def test_sample_conditional(session_tf, whiten, full_cov, full_output_cov):
    q_mu = np.random.randn(Data.M , Data.P)  # M x P
    q_sqrt = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)])  # P x M x M
    Z = Data.X[:Data.M, ...]  # M x D
    Xs = np.ones((Data.N, Data.D), dtype=float_type)

    feature = InducingPoints(Z.copy())
    kernel = RBF(Data.D)

    values = {"Z": Z, "Xnew": Xs, "q_mu": q_mu, "q_sqrt": q_sqrt}
    placeholders = _create_placeholder_dict(values)
    feed_dict = _create_feed_dict(placeholders, values)

    # Path 1
    sample_f = sample_conditional(placeholders["Xnew"], feature, kernel,
                                  placeholders["q_mu"], q_sqrt=placeholders["q_sqrt"], white=whiten,
                                  full_cov=full_cov, full_output_cov=full_output_cov, num_samples=int(1e5))
    value_f, mean_f, var_f = session_tf.run(sample_f, feed_dict=feed_dict)
    value_f = value_f.reshape((-1,) + value_f.shape[2:])

    # Path 2
    if full_output_cov:
        pytest.skip("sample_conditional with X instead of feature does not support full_output_cov")

    sample_x = sample_conditional(placeholders["Xnew"], placeholders["Z"], kernel,
                                  placeholders["q_mu"], q_sqrt=placeholders["q_sqrt"], white=whiten,
                                  full_cov=full_cov, full_output_cov=full_output_cov, num_samples=int(1e5))
    value_x, mean_x, var_x = session_tf.run(sample_x, feed_dict=feed_dict)
    value_x = value_x.reshape((-1,) + value_x.shape[2:])

    # check if mean and covariance of samples are similar
    np.testing.assert_array_almost_equal(np.mean(value_x, axis=0),
                                         np.mean(value_f, axis=0), decimal=1)
    np.testing.assert_array_almost_equal(np.cov(value_x, rowvar=False),
                                         np.cov(value_f, rowvar=False), decimal=1)
    np.testing.assert_allclose(mean_x, mean_f)
    np.testing.assert_allclose(var_x, var_f)


def test_sample_conditional_mixedkernel(session_tf):
    q_mu = np.random.randn(Data.M , Data.L)  # M x L
    q_sqrt = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.L)])  # L x M x M
    Z = Data.X[:Data.M,...]  # M x D
    N = int(10e5)
    Xs = np.ones((N, Data.D), dtype=float_type)


    values = {"Xnew": Xs, "q_mu": q_mu, "q_sqrt": q_sqrt}
    placeholders = _create_placeholder_dict(values)
    feed_dict = _create_feed_dict(placeholders, values)

    # Path 1: mixed kernel: most efficient route
    W = np.random.randn(Data.P, Data.L)
    mixed_kernel = mk.SeparateMixedMok([RBF(Data.D) for _ in range(Data.L)], W)
    mixed_feature = mf.MixedKernelSharedMof(InducingPoints(Z.copy()))

    sample = sample_conditional(placeholders["Xnew"], mixed_feature, mixed_kernel,
                                placeholders["q_mu"], q_sqrt=placeholders["q_sqrt"], white=True)
    value, mean, var = session_tf.run(sample, feed_dict=feed_dict)


    # Path 2: independent kernels, mixed later
    separate_kernel = mk.SeparateIndependentMok([RBF(Data.D) for _ in range(Data.L)])
    shared_feature = mf.SharedIndependentMof(InducingPoints(Z.copy()))
    sample2 = sample_conditional(placeholders["Xnew"], shared_feature, separate_kernel,
                                 placeholders["q_mu"], q_sqrt=placeholders["q_sqrt"], white=True)
    value2, mean2, var2 = session_tf.run(sample2, feed_dict=feed_dict)
    value2 = np.matmul(value2, W.T)
    # check if mean and covariance of samples are similar
    np.testing.assert_array_almost_equal(np.mean(value, axis=0),
                                         np.mean(value2, axis=0), decimal=1)
    np.testing.assert_array_almost_equal(np.cov(value, rowvar=False),
                                         np.cov(value2, rowvar=False), decimal=1)

# ------------------------------------------
# Test Mixed Mok Kgg
# ------------------------------------------

def test_MixedMok_Kgg(session_tf):
    data = DataMixedKernel
    kern_list = [RBF(data.D) for _ in range(data.L)]
    kern = mk.SeparateMixedMok(kern_list, W=data.W)

    Kgg = kern.compute_Kgg(Data.X, Data.X)  # L x N x N
    Kff = kern.compute_K(Data.X, Data.X)  # N x P x N x P

    # Kff = W @ Kgg @ W^T
    Kff_infered = np.einsum("lnm,pl,ql->npmq", Kgg, data.W, data.W)

    np.testing.assert_array_almost_equal(Kff, Kff_infered, decimal=5)







# ------------------------------------------
# Integration tests
# ------------------------------------------


def test_shared_independent_mok(session_tf):
    """
    In this test we use the same kernel and the same inducing features
    for each of the outputs. The outputs are considered to be uncorrelated.
    This is how GPflow handled multiple outputs before the multioutput framework was added.
    We compare three models here:
        1) an ineffient one, where we use a SharedIndepedentMok with InducingPoints.
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



def test_separate_independent_mok(session_tf):
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


def test_separate_independent_mof(session_tf):
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

    # Model 3 (Inefficient): an idenitical feature is used P times,
    # and treated as a separate feature.
    q_mu_3 = np.random.randn(Data.M, Data.P)
    q_sqrt_3 = np.array([np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)])  # P x M x M
    kern_list = [RBF(Data.D, variance=0.5, lengthscales=1.2)  for _ in range(Data.P)]
    kernel_3 = mk.SeparateIndependentMok(kern_list)
    feat_list_3 = [InducingPoints(Data.X[:Data.M, ...].copy()) for _ in range(Data.P)]
    feature_3 = mf.SeparateIndependentMof(feat_list_3)
    m3 = SVGP(Data.X, Data.Y, kernel_3, Gaussian(), feature_3, q_mu=q_mu_3, q_sqrt=q_sqrt_3)
    m3.set_trainable(False)
    m3.q_sqrt.set_trainable(True)
    m3.q_mu.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m3, maxiter=Data.MAXITER)

    check_equality_predictions(session_tf, [m1, m2, m3])


def test_mixed_mok_with_Id_vs_independent_mok(session_tf):
    data = DataMixedKernelWithEye
    # Independent model
    k1 = mk.SharedIndependentMok(RBF(data.D, variance=0.5, lengthscales=1.2), data.L)
    f1 = InducingPoints(data.X[:data.M, ...].copy())
    m1 = SVGP(data.X, data.Y, k1, Gaussian(), f1,
              q_mu=data.mu_data_full, q_sqrt=data.sqrt_data_full)
    m1.set_trainable(False)
    m1.q_sqrt.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m1, maxiter=data.MAXITER)

    # Mixed Model
    kern_list = [RBF(data.D, variance=0.5, lengthscales=1.2) for _ in range(data.L)]
    k2 = mk.SeparateMixedMok(kern_list, data.W)
    f2 = InducingPoints(data.X[:data.M, ...].copy())
    m2 = SVGP(data.X, data.Y, k2, Gaussian(), f2,
              q_mu=data.mu_data_full, q_sqrt=data.sqrt_data_full)
    m2.set_trainable(False)
    m2.q_sqrt.set_trainable(True)
    gpflow.training.ScipyOptimizer().minimize(m2, maxiter=data.MAXITER)

    check_equality_predictions(session_tf, [m1, m2])


def test_compare_mixed_kernel(session_tf):
    data = DataMixedKernel

    kern_list = [RBF(data.D) for _ in range(data.L)]
    k1 = mk.SeparateMixedMok(kern_list, W=data.W)
    f1 = mf.SharedIndependentMof(InducingPoints(data.X[:data.M,...].copy()))
    m1 = SVGP(data.X, data.Y, k1, Gaussian(), feat=f1, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    kern_list = [RBF(data.D) for _ in range(data.L)]
    k2 = mk.SeparateMixedMok(kern_list, W=data.W)
    f2 = mf.MixedKernelSharedMof(InducingPoints(data.X[:data.M,...].copy()))
    m2 = SVGP(data.X, data.Y, k2, Gaussian(), feat=f2, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    check_equality_predictions(session_tf, [m1, m2])


def test_multioutput_with_diag_q_sqrt(session_tf):
    data = DataMixedKernel

    q_sqrt_diag = np.ones((data.M, data.L)) * 2
    q_sqrt = np.repeat(np.eye(data.M)[None, ...], data.L, axis=0) * 2 # L x M x M

    kern_list = [RBF(data.D) for _ in range(data.L)]
    k1 = mk.SeparateMixedMok(kern_list, W=data.W)
    f1 = mf.SharedIndependentMof(InducingPoints(data.X[:data.M,...].copy()))
    m1 = SVGP(data.X, data.Y, k1, Gaussian(), feat=f1, q_mu=data.mu_data, q_sqrt=q_sqrt_diag, q_diag=True)

    kern_list = [RBF(data.D) for _ in range(data.L)]
    k2 = mk.SeparateMixedMok(kern_list, W=data.W)
    f2 = mf.SharedIndependentMof(InducingPoints(data.X[:data.M,...].copy()))
    m2 = SVGP(data.X, data.Y, k2, Gaussian(), feat=f2, q_mu=data.mu_data, q_sqrt=q_sqrt, q_diag=False)

    check_equality_predictions(session_tf, [m1, m2])

def test_MixedKernelSeparateMof(session_tf):
    data = DataMixedKernel

    kern_list = [RBF(data.D) for _ in range(data.L)]
    feat_list = [InducingPoints(data.X[:data.M, ...].copy()) for _ in range(data.L)]
    k1 = mk.SeparateMixedMok(kern_list, W=data.W)
    f1 = mf.SeparateIndependentMof(feat_list)
    m1 = SVGP(data.X, data.Y, k1, Gaussian(), feat=f1, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    kern_list = [RBF(data.D) for _ in range(data.L)]
    feat_list = [InducingPoints(data.X[:data.M, ...].copy()) for _ in range(data.L)]
    k2 = mk.SeparateMixedMok(kern_list, W=data.W)
    f2 = mf.MixedKernelSeparateMof(feat_list)
    m2 = SVGP(data.X, data.Y, k2, Gaussian(), feat=f2, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    check_equality_predictions(session_tf, [m1, m2])

