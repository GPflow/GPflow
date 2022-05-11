from typing import Callable, List, Sequence, Tuple, cast

import numpy as np
import pytest
import scipy
import tensorflow as tf
from _pytest.fixtures import SubRequest

import gpflow
import gpflow.inducing_variables.multioutput as mf
import gpflow.kernels.multioutput as mk
from gpflow import set_trainable
from gpflow.base import AnyNDArray, RegressionData
from gpflow.conditionals import sample_conditional
from gpflow.conditionals.util import (
    fully_correlated_conditional,
    fully_correlated_conditional_repeat,
    independent_interdomain_conditional,
    sample_mvn,
)
from gpflow.config import default_float, default_jitter
from gpflow.experimental.check_shapes import ShapeChecker, check_shapes
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import SquaredExponential
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP

float_type = default_float()
rng = np.random.RandomState(99201)

# ------------------------------------------
# Helpers
# ------------------------------------------


@check_shapes(
    "Xnew: [N, D]",
    "return[0]: [n_models, N, P]",
    "return[1]: [n_models, N, P, N, P] if full_cov and full_output_cov",
    "return[1]: [n_models, N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [n_models, P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [n_models, N, P] if (not full_cov) and (not full_output_cov)",
)
def predict_all(
    models: Sequence[SVGP], Xnew: tf.Tensor, full_cov: bool, full_output_cov: bool
) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    Returns the mean and variance of f(Xnew) for each model in `models`.
    """
    ms, vs = [], []
    for model in models:
        m, v = model.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        ms.append(m)
        vs.append(v)
    return ms, vs


@check_shapes(
    "arr[all]: [batch...]",
)
def assert_all_array_elements_almost_equal(arr: Sequence[tf.Tensor]) -> None:
    """
    Check if consecutive elements of `arr` are almost equal.
    """
    for i in range(len(arr) - 1):
        np.testing.assert_allclose(arr[i], arr[i + 1], atol=1e-5)


@check_shapes(
    "data[0]: [N, D]",
    "data[1]: [N, R]",
)
def check_equality_predictions(
    data: RegressionData, models: Sequence[SVGP], decimal: int = 3
) -> None:
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
    assert_all_array_elements_almost_equal(elbos)

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
    assert_all_array_elements_almost_equal(all_means)

    # check equality of all the variances within a category
    # (e.g. full_cov=True and full_output_cov=False)
    for var in [vars_tt, vars_tf, vars_ft, vars_ff]:
        assert_all_array_elements_almost_equal(var)

    # Here we check that the variance in different categories are equal
    # after transforming to the right shape.
    var_tt = vars_tt[0]  # N x P x N x P
    var_tf = vars_tf[0]  # P x N x c
    var_ft = vars_ft[0]  # N x P x P
    var_ff = vars_ff[0]  # N x P

    np.testing.assert_almost_equal(
        np.diagonal(var_tt, axis1=1, axis2=3),
        np.transpose(var_tf, [1, 2, 0]),
        decimal=decimal,
    )
    np.testing.assert_almost_equal(
        np.diagonal(var_tt, axis1=0, axis2=2),
        np.transpose(var_ft, [1, 2, 0]),
        decimal=decimal,
    )
    np.testing.assert_almost_equal(
        np.diagonal(np.diagonal(var_tt, axis1=0, axis2=2)), var_ff, decimal=decimal
    )


@check_shapes(
    "q_sqrt: [L, M, M]",
    "W: [L, L]",
    "return: [1, LM, LM]",
)
def expand_cov(q_sqrt: tf.Tensor, W: tf.Tensor) -> tf.Tensor:
    """
    :param q_sqrt: cholesky of covariance matrices
    :param W: mixing matrix
    :return: cholesky of the expanded covariance matrix
    """
    q_cov = np.matmul(q_sqrt, q_sqrt.transpose([0, 2, 1]))  # [L, M, M]
    q_cov_expanded = scipy.linalg.block_diag(*q_cov)  # [LM, LM]
    q_sqrt_expanded = np.linalg.cholesky(q_cov_expanded)  # [LM, LM]
    return q_sqrt_expanded[None, ...]


@check_shapes(
    "return: [L, M, M]",
)
def create_q_sqrt(M: int, L: int) -> AnyNDArray:
    """ returns an array of lower triangular matrices """
    return np.array([np.tril(rng.randn(M, M)) for _ in range(L)])  # [L, M, M]


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


class Data:
    cs = ShapeChecker().check_shape

    N, Ntest = 20, 5
    D = 1  # input dimension
    M = 3  # inducing points
    L = 2  # latent gps
    P = 3  # output dimension
    MAXITER = int(15e2)

    X = cs(tf.random.normal((N,), dtype=tf.float64)[:, None] * 10 - 5, "[N, 1]")
    G = cs(tf.concat([0.5 * tf.sin(3 * X) + X, 3.0 * tf.cos(X) - X], axis=1), "[N, L]")
    Ptrue = cs(tf.constant([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]], dtype=tf.float64), "[L, P]")
    Y = cs(G @ Ptrue, "[N, P]")
    Y += cs(tf.random.normal(Y.shape, dtype=tf.float64) * [0.2, 0.2, 0.2], "[N, P]")
    Xs = cs(tf.linspace(-6, 6, Ntest)[:, None], "[Ntest, 1]")
    data = (X, Y)


class DataMixedKernelWithEye(Data):
    """ Note in this class L == P """

    cs = ShapeChecker().check_shape

    M, L = 4, 3
    W = cs(tf.eye(L, dtype=tf.float64), "[L, L]")

    G = cs(
        tf.concat(
            [0.5 * tf.sin(3 * Data.X) + Data.X, 3.0 * tf.cos(Data.X) - Data.X, 1.0 + Data.X], axis=1
        ),
        "[N, P]",
    )

    mu_data = cs(tf.random.uniform((M, L), dtype=tf.float64), "[M, L]")
    sqrt_data = cs(tf.convert_to_tensor(create_q_sqrt(M, L)), "[L, M, M]")

    mu_data_full = cs(tf.reshape(mu_data @ W, [-1, 1]), "[LM, 1]")
    sqrt_data_full = cs(
        tf.convert_to_tensor(expand_cov(sqrt_data.numpy(), W.numpy())), "[1, LM, LM]"
    )

    Y = cs(G @ W, "[N, L]")
    Y += cs(
        tf.random.normal(Y.shape, dtype=tf.float64) * tf.ones((L,), dtype=tf.float64) * 0.2,
        "[N, L]",
    )
    data = (Data.X, Y)


class DataMixedKernel(Data):
    cs = ShapeChecker().check_shape

    M = 5
    L = 2
    P = 3
    W = cs(tf.convert_to_tensor(rng.randn(P, L)), "[P, L]")
    G = cs(
        tf.concat([0.5 * tf.sin(3 * Data.X) + Data.X, 3.0 * tf.cos(Data.X) - Data.X], axis=1),
        "[N, L]",
    )

    mu_data = cs(tf.random.normal((M, L), dtype=tf.float64), "[M, L]")
    sqrt_data = cs(create_q_sqrt(M, L), "[L, M, M]")

    Y = cs(G @ tf.transpose(W), "[N, P]")
    Y += cs(
        tf.random.normal(Y.shape, dtype=tf.float64) * tf.ones((P,), dtype=tf.float64) * 0.1,
        "[N, P]",
    )
    data = (Data.X, Y)


# ------------------------------------------
# Test sample conditional
# ------------------------------------------


def test_sample_mvn(full_cov: bool) -> None:
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


def test_sample_conditional(whiten: bool, full_cov: bool, full_output_cov: bool) -> None:
    if full_cov and full_output_cov:
        return

    q_mu = tf.random.uniform((Data.M, Data.P), dtype=tf.float64)  # [M, P]
    q_sqrt = tf.convert_to_tensor(
        [np.tril(tf.random.uniform((Data.M, Data.M), dtype=tf.float64)) for _ in range(Data.P)]
    )  # [P, M, M]

    Z = Data.X[: Data.M, ...]  # [M, D]
    Xs: AnyNDArray = np.ones((Data.N, Data.D), dtype=float_type)

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


def test_sample_conditional_mixedkernel() -> None:
    q_mu = tf.random.uniform((Data.M, Data.L), dtype=tf.float64)  # M x L
    q_sqrt = tf.convert_to_tensor(
        [np.tril(tf.random.uniform((Data.M, Data.M), dtype=tf.float64)) for _ in range(Data.L)]
    )  # L x M x M

    Z = Data.X[: Data.M, ...]  # M x D
    N = int(10e5)
    Xs: AnyNDArray = np.ones((N, Data.D), dtype=float_type)

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


QSqrtFactory = Callable[[tf.Tensor, int], tf.Tensor]


@pytest.fixture(
    name="fully_correlated_q_sqrt_factory",
    params=[lambda _, __: None, lambda LM, R: tf.eye(LM, batch_shape=(R,))],
)
def _q_sqrt_factory_fixture(request: SubRequest) -> QSqrtFactory:
    return cast(QSqrtFactory, request.param)


@pytest.mark.parametrize("R", [1, 2, 5])
def test_fully_correlated_conditional_repeat_shapes_fc_and_foc(
    R: int,
    fully_correlated_q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
) -> None:

    L, M, N, P = Data.L, Data.M, Data.N, Data.P

    Kmm = tf.ones((L * M, L * M)) + default_jitter() * tf.eye(L * M)
    Kmn = tf.ones((L * M, N, P))

    if full_cov and full_output_cov:
        Knn = tf.ones((N, P, N, P))
        expected_v_shape = [R, N, P, N, P]
    elif not full_cov and full_output_cov:
        Knn = tf.ones((N, P, P))
        expected_v_shape = [R, N, P, P]
    elif full_cov and not full_output_cov:
        Knn = tf.ones((P, N, N))
        expected_v_shape = [R, P, N, N]
    else:
        Knn = tf.ones((N, P))
        expected_v_shape = [R, N, P]

    f = tf.ones((L * M, R))
    q_sqrt = fully_correlated_q_sqrt_factory(L * M, R)

    m, v = fully_correlated_conditional_repeat(
        Kmn,
        Kmm,
        Knn,
        f,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
        q_sqrt=q_sqrt,
        white=whiten,
    )

    assert m.shape.as_list() == [R, N, P]
    assert v.shape.as_list() == expected_v_shape


def test_fully_correlated_conditional_repeat_whiten(whiten: bool) -> None:
    """
    This test checks the effect of the `white` flag, which changes the projection matrix `A`.

    The impact of the flag on the value of `A` can be easily verified by its effect on the
    predicted mean. While the predicted covariance is also a function of `A` this test does not
    inspect that value.
    """
    N, P = Data.N, Data.P

    Lm: AnyNDArray = np.random.randn(1, 1).astype(np.float32) ** 2
    Kmm = Lm * Lm + default_jitter()

    Kmn = tf.ones((1, N, P))

    Knn = tf.ones((N, P))
    f: AnyNDArray = np.random.randn(1, 1).astype(np.float32)

    mean, _ = fully_correlated_conditional_repeat(
        Kmn,
        Kmm,
        Knn,
        f,
        white=whiten,
    )

    if whiten:
        expected_mean = (f * Kmn) / Lm
    else:
        expected_mean = (f * Kmn) / Kmm

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-3)


def test_fully_correlated_conditional_shapes_fc_and_foc(
    fully_correlated_q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
) -> None:
    L, M, N, P = Data.L, Data.M, Data.N, Data.P

    Kmm = tf.ones((L * M, L * M)) + default_jitter() * tf.eye(L * M)
    Kmn = tf.ones((L * M, N, P))

    if full_cov and full_output_cov:
        Knn = tf.ones((N, P, N, P))
        expected_v_shape = [N, P, N, P]
    elif not full_cov and full_output_cov:
        Knn = tf.ones((N, P, P))
        expected_v_shape = [N, P, P]
    elif full_cov and not full_output_cov:
        Knn = tf.ones((P, N, N))
        expected_v_shape = [P, N, N]
    else:
        Knn = tf.ones((N, P))
        expected_v_shape = [N, P]

    f = tf.ones((L * M, 1))
    q_sqrt = fully_correlated_q_sqrt_factory(L * M, 1)

    m, v = fully_correlated_conditional(
        Kmn,
        Kmm,
        Knn,
        f,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
        q_sqrt=q_sqrt,
        white=whiten,
    )

    assert m.shape.as_list() == [N, P]
    assert v.shape.as_list() == expected_v_shape


# ------------------------------------------
# Test Mok Output Dims
# ------------------------------------------


def test_shapes_of_mok() -> None:
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


def test_MixedMok_Kgg() -> None:
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


def test_shared_independent_mok() -> None:
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
    q_sqrt_2: AnyNDArray = np.array(
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
    q_sqrt_3: AnyNDArray = np.array(
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


def test_separate_independent_mok() -> None:
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
        kernel_1,
        Gaussian(),
        inducing_variable_1,
        num_latent_gps=1,
        q_mu=q_mu_1,
        q_sqrt=q_sqrt_1,
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
    q_sqrt_2: AnyNDArray = np.array(
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


def test_separate_independent_mof() -> None:
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
    q_sqrt_2: AnyNDArray = np.array(
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
    q_sqrt_3: AnyNDArray = np.array(
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


def test_mixed_mok_with_Id_vs_independent_mok() -> None:
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


def test_compare_mixed_kernel() -> None:
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


def test_multioutput_with_diag_q_sqrt() -> None:
    data = DataMixedKernel

    q_sqrt_diag: AnyNDArray = np.ones((data.M, data.L)) * 2
    q_sqrt: AnyNDArray = np.repeat(np.eye(data.M)[None, ...], data.L, axis=0) * 2  # L x M x M

    kern_list = [SquaredExponential() for _ in range(data.L)]
    k1 = mk.LinearCoregionalization(kern_list, W=data.W)
    f1 = mf.SharedIndependentInducingVariables(InducingPoints(data.X[: data.M, ...]))
    model_1 = SVGP(
        k1,
        Gaussian(),
        inducing_variable=f1,
        q_mu=data.mu_data,
        q_sqrt=q_sqrt_diag,
        q_diag=True,
    )

    kern_list = [SquaredExponential() for _ in range(data.L)]
    k2 = mk.LinearCoregionalization(kern_list, W=data.W)
    f2 = mf.SharedIndependentInducingVariables(InducingPoints(data.X[: data.M, ...]))
    model_2 = SVGP(
        k2,
        Gaussian(),
        inducing_variable=f2,
        q_mu=data.mu_data,
        q_sqrt=q_sqrt,
        q_diag=False,
    )

    check_equality_predictions(Data.data, [model_1, model_2])


def test_MixedKernelSeparateMof() -> None:
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


def test_separate_independent_conditional_with_q_sqrt_none() -> None:
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

    gpflow.conditionals.conditional(
        data.X,
        inducing_variable,
        kernel,
        data.mu_data,
        full_cov=False,
        full_output_cov=False,
        q_sqrt=q_sqrt,
        white=True,
    )


def test_independent_interdomain_conditional_bug_regression() -> None:
    """
    Regression test for https://github.com/GPflow/GPflow/issues/818
    Not an exhaustive test
    """
    cs = ShapeChecker().check_shape

    M = 31
    N = 11
    D_lat = 5
    D_inp = D_lat * 7
    L = 2
    P = 3

    X = cs(np.random.randn(N, D_inp), "[N, D_inp]")
    Zs = cs([np.random.randn(M, D_lat) for _ in range(L)], "[L, M, D_lat]")
    k = gpflow.kernels.SquaredExponential(lengthscales=np.ones(D_lat))

    @check_shapes(
        "Z: [M, D_lat]",
        "X: [N, D_inp]",
        "return: [P, M, N]",
    )
    def compute_Kmn(Z: tf.Tensor, X: tf.Tensor) -> tf.Tensor:
        return tf.stack([k(Z, X[:, i * D_lat : (i + 1) * D_lat]) for i in range(P)])

    @check_shapes(
        "X: [N, D_inp]",
        "return: [P, N]",
    )
    def compute_Knn(X: tf.Tensor) -> tf.Tensor:
        return tf.stack([k(X[:, i * D_lat : (i + 1) * D_lat], full_cov=False) for i in range(P)])

    Kmm = cs(tf.stack([k(Z) for Z in Zs]), "[L, M, M]")
    Kmn = cs(tf.stack([compute_Kmn(Z, X) for Z in Zs]), "[L, P, M, N]")
    Kmn = cs(tf.transpose(Kmn, [2, 0, 3, 1]), "[M, L, N, P]")
    Knn = cs(tf.transpose(compute_Knn(X)), "[N, P]")
    q_mu = cs(tf.convert_to_tensor(np.zeros((M, L))), "[M, L]")
    q_sqrt = cs(tf.convert_to_tensor(np.stack([np.eye(M) for _ in range(L)])), "[L, M, M]")

    _, _ = independent_interdomain_conditional(
        Kmn, Kmm, Knn, q_mu, q_sqrt=q_sqrt, full_cov=False, full_output_cov=False
    )


def test_independent_interdomain_conditional_whiten(whiten: bool) -> None:
    """
    This test checks the effect of the `white` flag, which changes the projection matrix `A`.

    The impact of the flag on the value of `A` can be easily verified by its effect on the
    predicted mean. While the predicted covariance is also a function of `A` this test does not
    inspect that value.
    """
    N, P = Data.N, Data.P

    Lm: AnyNDArray = np.random.randn(1, 1, 1).astype(np.float32) ** 2
    Kmm = Lm * Lm + default_jitter()

    Kmn = tf.ones((1, 1, N, P))

    Knn = tf.ones((N, P))
    f: AnyNDArray = np.random.randn(1, 1).astype(np.float32)

    mean, _ = independent_interdomain_conditional(
        Kmn,
        Kmm,
        Knn,
        f,
        white=whiten,
    )

    if whiten:
        expected_mean = (f * Kmn) / Lm
    else:
        expected_mean = (f * Kmn) / Kmm

    np.testing.assert_allclose(mean, expected_mean[0][0], rtol=1e-2)
