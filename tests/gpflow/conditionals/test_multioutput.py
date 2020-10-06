import numpy as np
import pytest
import scipy
import tensorflow as tf

import gpflow
import gpflow.inducing_variables.multioutput as mf
import gpflow.kernels.multioutput as mk
from gpflow import set_trainable
from gpflow.conditionals import sample_conditional
from gpflow.conditionals.util import (
    fully_correlated_conditional,
    fully_correlated_conditional_repeat,
    independent_interdomain_conditional,
    sample_mvn,
)
from gpflow.config import default_float, default_jitter
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import SquaredExponential
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP

float_type = default_float()
rng = np.random.RandomState(99201)


# ------------------------------------------
# Helpers
# ------------------------------------------


def predict(model, Xnew, full_cov, full_output_cov):
    m, v = model.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
    return [m, v]


def predict_all(models, Xnew, full_cov, full_output_cov):
    """
    Returns the mean and variance of f(Xnew) for each model in `models`.
    """
    ms, vs = [], []
    for model in models:
        m, v = predict(model, Xnew, full_cov, full_output_cov)
        ms.append(m)
        vs.append(v)
    return ms, vs


def assert_all_array_elements_almost_equal(arr, decimal):
    """
    Check if consecutive elements of `arr` are almost equal.
    """
    for i in range(len(arr) - 1):
        np.testing.assert_allclose(arr[i], arr[i + 1], atol=1e-5)


def check_equality_predictions(data, models, decimal=3):
    """
    Executes a couple of checks to compare the equality of predictions
    of different models. The models should be configured with the same
    training data (X, Y). The following checks are done:
    - check if elbo is (almost) equal for all models
    - check if predicted mean is (almost) equal
    - check if predicted variance is (almost) equal.
      All possible variances over the inputs and outputs are calculated
      and equality is checked.
    - check if variances within model are consistent. Parts of the covariance
      matrices should overlap, and this is tested.
    """

    elbos = [m.elbo(data) for m in models]

    # Check equality of log likelihood
    assert_all_array_elements_almost_equal(elbos, decimal=5)

    # Predict: full_cov = True and full_output_cov = True
    means_tt, vars_tt = predict_all(models, Data.Xs, full_cov=True, full_output_cov=True)
    # Predict: full_cov = True and full_output_cov = False
    means_tf, vars_tf = predict_all(models, Data.Xs, full_cov=True, full_output_cov=False)
    # Predict: full_cov = False and full_output_cov = True
    means_ft, vars_ft = predict_all(models, Data.Xs, full_cov=False, full_output_cov=True)
    # Predict: full_cov = False and full_output_cov = False
    means_ff, vars_ff = predict_all(models, Data.Xs, full_cov=False, full_output_cov=False)

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

    np.testing.assert_almost_equal(
        np.diagonal(var_tt, axis1=1, axis2=3), np.transpose(var_tf, [1, 2, 0]), decimal=decimal,
    )
    np.testing.assert_almost_equal(
        np.diagonal(var_tt, axis1=0, axis2=2), np.transpose(var_ft, [1, 2, 0]), decimal=decimal,
    )
    np.testing.assert_almost_equal(
        np.diagonal(np.diagonal(var_tt, axis1=0, axis2=2)), var_ff, decimal=decimal
    )


def expand_cov(q_sqrt, W):
    """
    :param G: cholesky of covariance matrices, L x M x M
    :param W: mixing matrix (square),  L x L
    :return: cholesky of 1 x LM x LM covariance matrix
    """
    q_cov = np.matmul(q_sqrt, q_sqrt.transpose([0, 2, 1]))  # [L, M, M]
    q_cov_expanded = scipy.linalg.block_diag(*q_cov)  # [LM, LM]
    q_sqrt_expanded = np.linalg.cholesky(q_cov_expanded)  # [LM, LM]
    return q_sqrt_expanded[None, ...]


def create_q_sqrt(M, L):
    """ returns an array of L lower triangular matrices of size M x M """
    return np.array([np.tril(rng.randn(M, M)) for _ in range(L)])  # [L, M, M]


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
    X = tf.random.normal((N,), dtype=tf.float64)[:, None] * 10 - 5
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))
    Ptrue = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # [L, P]

    Y = tf.convert_to_tensor(G @ Ptrue)
    G = tf.convert_to_tensor(np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X)))
    Ptrue = tf.convert_to_tensor(np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]]))  # [L, P]
    Y += tf.random.normal(Y.shape, dtype=tf.float64) * [0.2, 0.2, 0.2]
    Xs = tf.convert_to_tensor(np.linspace(-6, 6, Ntest)[:, None])
    data = (X, Y)


class DataMixedKernelWithEye(Data):
    """ Note in this class L == P """

    M, L = 4, 3
    W = np.eye(L)

    G = np.hstack(
        [0.5 * np.sin(3 * Data.X) + Data.X, 3.0 * np.cos(Data.X) - Data.X, 1.0 + Data.X]
    )  # [N, P]

    mu_data = tf.random.uniform((M, L), dtype=tf.float64)  # [M, L]
    sqrt_data = create_q_sqrt(M, L)  # [L, M, M]

    mu_data_full = tf.reshape(mu_data @ W, [-1, 1])  # [L, 1]
    sqrt_data_full = expand_cov(sqrt_data, W)  # [1, LM, LM]

    Y = tf.convert_to_tensor(G @ W)
    G = tf.convert_to_tensor(G)
    W = tf.convert_to_tensor(W)
    sqrt_data = tf.convert_to_tensor(sqrt_data)
    sqrt_data_full = tf.convert_to_tensor(sqrt_data_full)
    Y += tf.random.normal(Y.shape, dtype=tf.float64) * tf.ones((L,), dtype=tf.float64) * 0.2
    data = (Data.X, Y)


class DataMixedKernel(Data):
    M = 5
    L = 2
    P = 3
    W = rng.randn(P, L)
    G = np.hstack([0.5 * np.sin(3 * Data.X) + Data.X, 3.0 * np.cos(Data.X) - Data.X])  # [N, L]

    mu_data = tf.random.normal((M, L), dtype=tf.float64)  # [M, L]
    sqrt_data = create_q_sqrt(M, L)  # [L, M, M]

    Y = tf.convert_to_tensor(G @ W.T)
    G = tf.convert_to_tensor(G)
    W = tf.convert_to_tensor(W)
    sqrt_data = tf.convert_to_tensor(sqrt_data)
    Y += tf.random.normal(Y.shape, dtype=tf.float64) * tf.ones((P,), dtype=tf.float64) * 0.1
    data = (Data.X, Y)


# ------------------------------------------
# Test sample conditional
# ------------------------------------------


@pytest.mark.parametrize("full_cov", [True, False])
def test_sample_mvn(full_cov):
    """
    Draws 10,000 samples from a distribution
    with known mean and covariance. The test checks
    if the mean and covariance of the samples is
    close to the true mean and covariance.
    """
    N, D = 10000, 2
    means = tf.ones((N, D), dtype=float_type)
    if full_cov:
        covs = tf.eye(D, batch_shape=[N], dtype=float_type)
    else:
        covs = tf.ones((N, D), dtype=float_type)

    samples = sample_mvn(means, covs, full_cov)
    samples_mean = np.mean(samples, axis=0)
    samples_cov = np.cov(samples, rowvar=False)

    np.testing.assert_array_almost_equal(samples_mean, [1.0, 1.0], decimal=1)
    np.testing.assert_array_almost_equal(samples_cov, [[1.0, 0.0], [0.0, 1.0]], decimal=1)


@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_sample_conditional(whiten, full_cov, full_output_cov):
    if full_cov and full_output_cov:
        return

    q_mu = tf.random.uniform((Data.M, Data.P), dtype=tf.float64)  # [M, P]
    q_sqrt = tf.convert_to_tensor(
        [np.tril(tf.random.uniform((Data.M, Data.M), dtype=tf.float64)) for _ in range(Data.P)]
    )  # [P, M, M]

    Z = Data.X[: Data.M, ...]  # [M, D]
    Xs = np.ones((Data.N, Data.D), dtype=float_type)

    inducing_variable = InducingPoints(Z)
    kernel = SquaredExponential()

    # Path 1
    value_f, mean_f, var_f = sample_conditional(
        Xs,
        inducing_variable,
        kernel,
        q_mu,
        q_sqrt=q_sqrt,
        white=whiten,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
        num_samples=int(1e5),
    )
    value_f = value_f.numpy().reshape((-1,) + value_f.numpy().shape[2:])

    # Path 2
    if full_output_cov:
        pytest.skip(
            "sample_conditional with X instead of inducing_variable does not support full_output_cov"
        )

    value_x, mean_x, var_x = sample_conditional(
        Xs,
        Z,
        kernel,
        q_mu,
        q_sqrt=q_sqrt,
        white=whiten,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
        num_samples=int(1e5),
    )
    value_x = value_x.numpy().reshape((-1,) + value_x.numpy().shape[2:])

    # check if mean and covariance of samples are similar
    np.testing.assert_array_almost_equal(
        np.mean(value_x, axis=0), np.mean(value_f, axis=0), decimal=1
    )
    np.testing.assert_array_almost_equal(
        np.cov(value_x, rowvar=False), np.cov(value_f, rowvar=False), decimal=1
    )
    np.testing.assert_allclose(mean_x, mean_f)
    np.testing.assert_allclose(var_x, var_f)


def test_sample_conditional_mixedkernel():
    q_mu = tf.random.uniform((Data.M, Data.L), dtype=tf.float64)  # M x L
    q_sqrt = tf.convert_to_tensor(
        [np.tril(tf.random.uniform((Data.M, Data.M), dtype=tf.float64)) for _ in range(Data.L)]
    )  # L x M x M

    Z = Data.X[: Data.M, ...]  # M x D
    N = int(10e5)
    Xs = np.ones((N, Data.D), dtype=float_type)

    # Path 1: mixed kernel: most efficient route
    W = np.random.randn(Data.P, Data.L)
    mixed_kernel = mk.LinearCoregionalization([SquaredExponential() for _ in range(Data.L)], W)
    optimal_inducing_variable = mf.SharedIndependentInducingVariables(InducingPoints(Z))

    value, mean, var = sample_conditional(
        Xs, optimal_inducing_variable, mixed_kernel, q_mu, q_sqrt=q_sqrt, white=True
    )

    # Path 2: independent kernels, mixed later
    separate_kernel = mk.SeparateIndependent([SquaredExponential() for _ in range(Data.L)])
    fallback_inducing_variable = mf.SharedIndependentInducingVariables(InducingPoints(Z))

    value2, mean2, var2 = sample_conditional(
        Xs, fallback_inducing_variable, separate_kernel, q_mu, q_sqrt=q_sqrt, white=True
    )
    value2 = np.matmul(value2, W.T)
    # check if mean and covariance of samples are similar
    np.testing.assert_array_almost_equal(np.mean(value, axis=0), np.mean(value2, axis=0), decimal=1)
    np.testing.assert_array_almost_equal(
        np.cov(value, rowvar=False), np.cov(value2, rowvar=False), decimal=1
    )


@pytest.mark.parametrize(
    "func, R",
    [
        (fully_correlated_conditional_repeat, 5),
        (fully_correlated_conditional_repeat, 1),
        (fully_correlated_conditional, 1),
    ],
)
def test_fully_correlated_conditional_repeat_shapes(func, R):
    L, M, N, P = Data.L, Data.M, Data.N, Data.P

    Kmm = tf.ones((L * M, L * M)) + default_jitter() * tf.eye(L * M)
    Kmn = tf.ones((L * M, N, P))
    Knn = tf.ones((N, P))
    f = tf.ones((L * M, R))
    q_sqrt = None
    white = True

    m, v = func(
        Kmn, Kmm, Knn, f, full_cov=False, full_output_cov=False, q_sqrt=q_sqrt, white=white,
    )

    assert v.shape.as_list() == m.shape.as_list()


# ------------------------------------------
# Test Mok Output Dims
# ------------------------------------------


def test_shapes_of_mok():
    data = DataMixedKernel

    kern_list = [SquaredExponential() for _ in range(data.L)]

    k1 = mk.LinearCoregionalization(kern_list, W=data.W)
    assert k1.num_latent_gps == data.L

    k2 = mk.SeparateIndependent(kern_list)
    assert k2.num_latent_gps == data.L

    dims = 5
    k3 = mk.SharedIndependent(SquaredExponential(), dims)
    assert k3.num_latent_gps == dims


# ------------------------------------------
# Test Mixed Mok Kgg
# ------------------------------------------


def test_MixedMok_Kgg():
    data = DataMixedKernel
    kern_list = [SquaredExponential() for _ in range(data.L)]
    kernel = mk.LinearCoregionalization(kern_list, W=data.W)

    Kgg = kernel.Kgg(Data.X, Data.X)  # L x N x N
    Kff = kernel.K(Data.X, Data.X)  # N x P x N x P

    # Kff = W @ Kgg @ W^T
    Kff_infered = np.einsum("lnm,pl,ql->npmq", Kgg, data.W, data.W)

    np.testing.assert_array_almost_equal(Kff, Kff_infered, decimal=5)


# ------------------------------------------
# Integration tests
# ------------------------------------------


def test_shared_independent_mok():
    """
    In this test we use the same kernel and the same inducing inducing
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
    np.random.seed(0)
    # Model 1
    q_mu_1 = np.random.randn(Data.M * Data.P, 1)  # MP x 1
    q_sqrt_1 = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP
    kernel_1 = mk.SharedIndependent(SquaredExponential(variance=0.5, lengthscales=1.2), Data.P)
    inducing_variable = InducingPoints(Data.X[: Data.M, ...])
    model_1 = SVGP(
        kernel_1,
        Gaussian(),
        inducing_variable,
        q_mu=q_mu_1,
        q_sqrt=q_sqrt_1,
        num_latent_gps=Data.Y.shape[-1],
    )
    set_trainable(model_1, False)
    set_trainable(model_1.q_sqrt, True)

    gpflow.optimizers.Scipy().minimize(
        model_1.training_loss_closure(Data.data),
        variables=model_1.trainable_variables,
        options=dict(maxiter=500),
        method="BFGS",
        compile=True,
    )

    # Model 2
    q_mu_2 = np.reshape(q_mu_1, [Data.M, Data.P])  # M x P
    q_sqrt_2 = np.array(
        [np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)]
    )  # P x M x M
    kernel_2 = SquaredExponential(variance=0.5, lengthscales=1.2)
    inducing_variable_2 = InducingPoints(Data.X[: Data.M, ...])
    model_2 = SVGP(
        kernel_2,
        Gaussian(),
        inducing_variable_2,
        num_latent_gps=Data.P,
        q_mu=q_mu_2,
        q_sqrt=q_sqrt_2,
    )
    set_trainable(model_2, False)
    set_trainable(model_2.q_sqrt, True)

    gpflow.optimizers.Scipy().minimize(
        model_2.training_loss_closure(Data.data),
        variables=model_2.trainable_variables,
        options=dict(maxiter=500),
        method="BFGS",
        compile=True,
    )

    # Model 3
    q_mu_3 = np.reshape(q_mu_1, [Data.M, Data.P])  # M x P
    q_sqrt_3 = np.array(
        [np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)]
    )  # P x M x M
    kernel_3 = mk.SharedIndependent(SquaredExponential(variance=0.5, lengthscales=1.2), Data.P)
    inducing_variable_3 = mf.SharedIndependentInducingVariables(
        InducingPoints(Data.X[: Data.M, ...])
    )
    model_3 = SVGP(
        kernel_3,
        Gaussian(),
        inducing_variable_3,
        num_latent_gps=Data.P,
        q_mu=q_mu_3,
        q_sqrt=q_sqrt_3,
    )
    set_trainable(model_3, False)
    set_trainable(model_3.q_sqrt, True)

    gpflow.optimizers.Scipy().minimize(
        model_3.training_loss_closure(Data.data),
        variables=model_3.trainable_variables,
        options=dict(maxiter=500),
        method="BFGS",
        compile=True,
    )

    check_equality_predictions(Data.data, [model_1, model_2, model_3])


def test_separate_independent_mok():
    """
    We use different independent kernels for each of the output dimensions.
    We can achieve this in two ways:
        1) efficient: SeparateIndependentMok with Shared/SeparateIndependentMof
        2) inefficient: SeparateIndependentMok with InducingPoints
    However, both methods should return the same conditional,
    and after optimization return the same log likelihood.
    """
    # Model 1 (Inefficient)
    q_mu_1 = np.random.randn(Data.M * Data.P, 1)
    q_sqrt_1 = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP

    kern_list_1 = [SquaredExponential(variance=0.5, lengthscales=1.2) for _ in range(Data.P)]
    kernel_1 = mk.SeparateIndependent(kern_list_1)
    inducing_variable_1 = InducingPoints(Data.X[: Data.M, ...])
    model_1 = SVGP(
        kernel_1, Gaussian(), inducing_variable_1, num_latent_gps=1, q_mu=q_mu_1, q_sqrt=q_sqrt_1,
    )
    set_trainable(model_1, False)
    set_trainable(model_1.q_sqrt, True)
    set_trainable(model_1.q_mu, True)

    gpflow.optimizers.Scipy().minimize(
        model_1.training_loss_closure(Data.data),
        variables=model_1.trainable_variables,
        method="BFGS",
        compile=True,
    )

    # Model 2 (efficient)
    q_mu_2 = np.random.randn(Data.M, Data.P)
    q_sqrt_2 = np.array(
        [np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)]
    )  # P x M x M
    kern_list_2 = [SquaredExponential(variance=0.5, lengthscales=1.2) for _ in range(Data.P)]
    kernel_2 = mk.SeparateIndependent(kern_list_2)
    inducing_variable_2 = mf.SharedIndependentInducingVariables(
        InducingPoints(Data.X[: Data.M, ...])
    )
    model_2 = SVGP(
        kernel_2,
        Gaussian(),
        inducing_variable_2,
        num_latent_gps=Data.P,
        q_mu=q_mu_2,
        q_sqrt=q_sqrt_2,
    )
    set_trainable(model_2, False)
    set_trainable(model_2.q_sqrt, True)
    set_trainable(model_2.q_mu, True)

    gpflow.optimizers.Scipy().minimize(
        model_2.training_loss_closure(Data.data),
        variables=model_2.trainable_variables,
        method="BFGS",
        compile=True,
    )

    check_equality_predictions(Data.data, [model_1, model_2])


def test_separate_independent_mof():
    """
    Same test as above but we use different (i.e. separate) inducing inducing
    for each of the output dimensions.
    """
    np.random.seed(0)

    # Model 1 (INefficient)
    q_mu_1 = np.random.randn(Data.M * Data.P, 1)
    q_sqrt_1 = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP

    kernel_1 = mk.SharedIndependent(SquaredExponential(variance=0.5, lengthscales=1.2), Data.P)
    inducing_variable_1 = InducingPoints(Data.X[: Data.M, ...])
    model_1 = SVGP(kernel_1, Gaussian(), inducing_variable_1, q_mu=q_mu_1, q_sqrt=q_sqrt_1)
    set_trainable(model_1, False)
    set_trainable(model_1.q_sqrt, True)
    set_trainable(model_1.q_mu, True)

    gpflow.optimizers.Scipy().minimize(
        model_1.training_loss_closure(Data.data),
        variables=model_1.trainable_variables,
        method="BFGS",
        compile=True,
    )

    # Model 2 (efficient)
    q_mu_2 = np.random.randn(Data.M, Data.P)
    q_sqrt_2 = np.array(
        [np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)]
    )  # P x M x M
    kernel_2 = mk.SharedIndependent(SquaredExponential(variance=0.5, lengthscales=1.2), Data.P)
    inducing_variable_list_2 = [InducingPoints(Data.X[: Data.M, ...]) for _ in range(Data.P)]
    inducing_variable_2 = mf.SeparateIndependentInducingVariables(inducing_variable_list_2)
    model_2 = SVGP(kernel_2, Gaussian(), inducing_variable_2, q_mu=q_mu_2, q_sqrt=q_sqrt_2)
    set_trainable(model_2, False)
    set_trainable(model_2.q_sqrt, True)
    set_trainable(model_2.q_mu, True)

    gpflow.optimizers.Scipy().minimize(
        model_2.training_loss_closure(Data.data),
        variables=model_2.trainable_variables,
        method="BFGS",
        compile=True,
    )

    # Model 3 (Inefficient): an idenitical inducing variable is used P times,
    # and treated as a separate one.
    q_mu_3 = np.random.randn(Data.M, Data.P)
    q_sqrt_3 = np.array(
        [np.tril(np.random.randn(Data.M, Data.M)) for _ in range(Data.P)]
    )  # P x M x M
    kern_list = [SquaredExponential(variance=0.5, lengthscales=1.2) for _ in range(Data.P)]
    kernel_3 = mk.SeparateIndependent(kern_list)
    inducing_variable_list_3 = [InducingPoints(Data.X[: Data.M, ...]) for _ in range(Data.P)]
    inducing_variable_3 = mf.SeparateIndependentInducingVariables(inducing_variable_list_3)
    model_3 = SVGP(kernel_3, Gaussian(), inducing_variable_3, q_mu=q_mu_3, q_sqrt=q_sqrt_3)
    set_trainable(model_3, False)
    set_trainable(model_3.q_sqrt, True)
    set_trainable(model_3.q_mu, True)

    gpflow.optimizers.Scipy().minimize(
        model_3.training_loss_closure(Data.data),
        variables=model_3.trainable_variables,
        method="BFGS",
        compile=True,
    )

    check_equality_predictions(Data.data, [model_1, model_2, model_3])


def test_mixed_mok_with_Id_vs_independent_mok():
    data = DataMixedKernelWithEye
    # Independent model
    k1 = mk.SharedIndependent(SquaredExponential(variance=0.5, lengthscales=1.2), data.L)
    f1 = InducingPoints(data.X[: data.M, ...])
    model_1 = SVGP(k1, Gaussian(), f1, q_mu=data.mu_data_full, q_sqrt=data.sqrt_data_full)
    set_trainable(model_1, False)
    set_trainable(model_1.q_sqrt, True)

    gpflow.optimizers.Scipy().minimize(
        model_1.training_loss_closure(Data.data),
        variables=model_1.trainable_variables,
        method="BFGS",
        compile=True,
    )

    # Mixed Model
    kern_list = [SquaredExponential(variance=0.5, lengthscales=1.2) for _ in range(data.L)]
    k2 = mk.LinearCoregionalization(kern_list, data.W)
    f2 = InducingPoints(data.X[: data.M, ...])
    model_2 = SVGP(k2, Gaussian(), f2, q_mu=data.mu_data_full, q_sqrt=data.sqrt_data_full)
    set_trainable(model_2, False)
    set_trainable(model_2.q_sqrt, True)

    gpflow.optimizers.Scipy().minimize(
        model_2.training_loss_closure(Data.data),
        variables=model_2.trainable_variables,
        method="BFGS",
        compile=True,
    )

    check_equality_predictions(Data.data, [model_1, model_2])


def test_compare_mixed_kernel():
    data = DataMixedKernel

    kern_list = [SquaredExponential() for _ in range(data.L)]
    k1 = mk.LinearCoregionalization(kern_list, W=data.W)
    f1 = mf.SharedIndependentInducingVariables(InducingPoints(data.X[: data.M, ...]))
    model_1 = SVGP(k1, Gaussian(), inducing_variable=f1, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    kern_list = [SquaredExponential() for _ in range(data.L)]
    k2 = mk.LinearCoregionalization(kern_list, W=data.W)
    f2 = mf.SharedIndependentInducingVariables(InducingPoints(data.X[: data.M, ...]))
    model_2 = SVGP(k2, Gaussian(), inducing_variable=f2, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    check_equality_predictions(Data.data, [model_1, model_2])


def test_multioutput_with_diag_q_sqrt():
    data = DataMixedKernel

    q_sqrt_diag = np.ones((data.M, data.L)) * 2
    q_sqrt = np.repeat(np.eye(data.M)[None, ...], data.L, axis=0) * 2  # L x M x M

    kern_list = [SquaredExponential() for _ in range(data.L)]
    k1 = mk.LinearCoregionalization(kern_list, W=data.W)
    f1 = mf.SharedIndependentInducingVariables(InducingPoints(data.X[: data.M, ...]))
    model_1 = SVGP(
        k1, Gaussian(), inducing_variable=f1, q_mu=data.mu_data, q_sqrt=q_sqrt_diag, q_diag=True,
    )

    kern_list = [SquaredExponential() for _ in range(data.L)]
    k2 = mk.LinearCoregionalization(kern_list, W=data.W)
    f2 = mf.SharedIndependentInducingVariables(InducingPoints(data.X[: data.M, ...]))
    model_2 = SVGP(
        k2, Gaussian(), inducing_variable=f2, q_mu=data.mu_data, q_sqrt=q_sqrt, q_diag=False,
    )

    check_equality_predictions(Data.data, [model_1, model_2])


def test_MixedKernelSeparateMof():
    data = DataMixedKernel

    kern_list = [SquaredExponential() for _ in range(data.L)]
    inducing_variable_list = [InducingPoints(data.X[: data.M, ...]) for _ in range(data.L)]
    k1 = mk.LinearCoregionalization(kern_list, W=data.W)
    f1 = mf.SeparateIndependentInducingVariables(inducing_variable_list)
    model_1 = SVGP(k1, Gaussian(), inducing_variable=f1, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    kern_list = [SquaredExponential() for _ in range(data.L)]
    inducing_variable_list = [InducingPoints(data.X[: data.M, ...]) for _ in range(data.L)]
    k2 = mk.LinearCoregionalization(kern_list, W=data.W)
    f2 = mf.SeparateIndependentInducingVariables(inducing_variable_list)
    model_2 = SVGP(k2, Gaussian(), inducing_variable=f2, q_mu=data.mu_data, q_sqrt=data.sqrt_data)

    check_equality_predictions(Data.data, [model_1, model_2])


def test_separate_independent_conditional_with_q_sqrt_none():
    """
    In response to bug #1523, this test checks that separate_independent_condtional
    does not fail when q_sqrt=None.
    """
    q_sqrt = None
    data = DataMixedKernel

    kern_list = [SquaredExponential() for _ in range(data.L)]
    kernel = gpflow.kernels.SeparateIndependent(kern_list)
    inducing_variable_list = [InducingPoints(data.X[: data.M, ...]) for _ in range(data.L)]
    inducing_variable = mf.SeparateIndependentInducingVariables(inducing_variable_list)

    mu_1, var_1 = gpflow.conditionals.conditional(
        data.X,
        inducing_variable,
        kernel,
        data.mu_data,
        full_cov=False,
        full_output_cov=False,
        q_sqrt=q_sqrt,
        white=True,
    )


def test_independent_interdomain_conditional_bug_regression():
    """
    Regression test for https://github.com/GPflow/GPflow/issues/818
    Not an exhaustive test
    """
    M = 31
    N = 11
    D_lat = 5
    D_inp = D_lat * 7
    L = 2
    P = 3

    X = np.random.randn(N, D_inp)
    Zs = [np.random.randn(M, D_lat) for _ in range(L)]
    k = gpflow.kernels.SquaredExponential(lengthscales=np.ones(D_lat))

    def compute_Kmn(Z, X):
        return tf.stack([k(Z, X[:, i * D_lat : (i + 1) * D_lat]) for i in range(P)])

    def compute_Knn(X):
        return tf.stack([k(X[:, i * D_lat : (i + 1) * D_lat], full_cov=False) for i in range(P)])

    Kmm = tf.stack([k(Z) for Z in Zs])  # L x M x M
    Kmn = tf.stack([compute_Kmn(Z, X) for Z in Zs])  # L x P x M x N
    Kmn = tf.transpose(Kmn, [2, 0, 3, 1])  # -> M x L x N x P
    Knn = tf.transpose(compute_Knn(X))  # N x P
    q_mu = tf.convert_to_tensor(np.zeros((M, L)))
    q_sqrt = tf.convert_to_tensor(np.stack([np.eye(M) for _ in range(L)]))
    tf.debugging.assert_shapes(
        [
            (Kmm, ["L", "M", "M"]),
            (Kmn, ["M", "L", "N", "P"]),
            (Knn, ["N", "P"]),
            (q_mu, ["M", "L"]),
            (q_sqrt, ["L", "M", "M"]),
        ]
    )

    _, _ = independent_interdomain_conditional(
        Kmn, Kmm, Knn, q_mu, q_sqrt=q_sqrt, full_cov=False, full_output_cov=False
    )
