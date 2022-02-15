import numpy as np
import pytest
from numpy.testing import assert_allclose

import gpflow
from tests.gpflow.kernels.reference import ref_changepoints

rng = np.random.RandomState(1)


@pytest.mark.parametrize(
    "locations, steepness, error_msg",
    [
        # 1. Kernels locations dimension mismatch
        [
            [1.0],
            1.0,
            r"Number of kernels \(3\) must be one more than the number of changepoint locations \(1\)",
        ],
        # 2. Locations steepness dimension mismatch
        [
            [1.0, 2.0],
            [1.0],
            r"Dimension of steepness \(1\) does not match number of changepoint locations \(2\)",
        ],
    ],
)
def test_changepoints_init_fail(locations, steepness, error_msg):
    kernels = [
        gpflow.kernels.Matern12(),
        gpflow.kernels.Linear(),
        gpflow.kernels.Matern32(),
    ]
    with pytest.raises(ValueError, match=error_msg):
        gpflow.kernels.ChangePoints(kernels, locations, steepness)


def _assert_changepoints_kern_err(X, kernels, locations, steepness):
    kernel = gpflow.kernels.ChangePoints(kernels, locations, steepness=steepness)
    reference_gram_matrix = ref_changepoints(X, kernels, locations, steepness)

    assert_allclose(kernel(X), reference_gram_matrix)
    assert_allclose(kernel.K_diag(X), np.diag(reference_gram_matrix))


@pytest.mark.parametrize("N", [2, 10])
@pytest.mark.parametrize(
    "kernels, locations, steepness",
    [
        # 1. Single changepoint
        [[gpflow.kernels.Constant(), gpflow.kernels.Constant()], [2.0], 5.0],
        # 2. Two changepoints
        [
            [
                gpflow.kernels.Constant(),
                gpflow.kernels.Constant(),
                gpflow.kernels.Constant(),
            ],
            [1.0, 2.0],
            5.0,
        ],
        # 3. Multiple steepness
        [
            [
                gpflow.kernels.Constant(),
                gpflow.kernels.Constant(),
                gpflow.kernels.Constant(),
            ],
            [1.0, 2.0],
            [5.0, 10.0],
        ],
        # 4. Variety of kernels
        [
            [
                gpflow.kernels.Matern12(),
                gpflow.kernels.Linear(),
                gpflow.kernels.SquaredExponential(),
                gpflow.kernels.Constant(),
            ],
            [1.0, 2.0, 3.0],
            5.0,
        ],
    ],
)
def test_changepoint_output(N, kernels, locations, steepness):
    X_data = rng.randn(N, 1)
    _assert_changepoints_kern_err(X_data, kernels, locations, steepness)


def test_changepoint_with_X1_X2():
    N = 100
    X = np.linspace(0, 100, N).reshape(N, 1)
    base_k1 = gpflow.kernels.Matern32(lengthscales=0.2)
    base_k2 = gpflow.kernels.Matern32(lengthscales=2.0)
    k = gpflow.kernels.ChangePoints([base_k1, base_k2], [0.0], steepness=5.0)
    K = k(X)
    assert K.shape == [N, N]

    N2 = 25
    X2 = np.linspace(0, 50, N2).reshape(N2, 1)
    K = k(X, X2)
    assert K.shape == [N, N2]


@pytest.mark.parametrize("switch_dim", [0, 1])
def test_changepoint_xslice_sigmoid(switch_dim):
    """
    Test shaping and slicing of X introduced to accommodate switch_dim parameter.
    """
    X = rng.rand(10, 2)
    locations = [2.0]
    steepness = 5.0

    X1 = X[:, [switch_dim]]
    sig_X1 = 1.0 / (1.0 + np.exp(-steepness * (X1[:, :, None] - locations)))

    Xslice = X[:, switch_dim].reshape(-1, 1, 1)
    sig_Xslice = 1.0 / (1.0 + np.exp(-steepness * (Xslice - locations)))

    assert_allclose(sig_X1, sig_Xslice)


@pytest.mark.parametrize("switch_dim", [0, 1])
def test_changepoint_xslice(switch_dim):
    """
    Test switch_dim behaviour in comparison to slicing on input X.
    """
    N, D = 10, 2
    locations = [2.0]
    steepness = 5.0
    X = rng.randn(N, D)
    RBF = gpflow.kernels.SquaredExponential

    kernel = gpflow.kernels.ChangePoints(
        [RBF(active_dims=[switch_dim]), RBF(active_dims=[switch_dim])],
        locations,
        steepness=steepness,
        switch_dim=switch_dim,
    )
    reference_gram_matrix = ref_changepoints(
        X[:, [switch_dim]], [RBF(), RBF()], locations, steepness
    )

    assert_allclose(kernel(X), reference_gram_matrix)


@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("switch_dim", [0, 1])
@pytest.mark.parametrize("active_dim", [0, 1])
def test_changepoint_ndim(D, switch_dim, active_dim):
    """
    Test Changepoints with varying combinations of switch_dim and active_dim.
    """
    N = 10
    X = rng.randn(N, D)
    RBF = gpflow.kernels.SquaredExponential
    locations = [2.0]
    steepness = 5.0

    kernel = gpflow.kernels.ChangePoints(
        [RBF(active_dims=[active_dim]), RBF(active_dims=[active_dim])],
        locations,
        steepness=steepness,
        switch_dim=switch_dim,
    )
    reference_gram_matrix = ref_changepoints(
        X,
        [RBF(active_dims=[active_dim]), RBF(active_dims=[active_dim])],
        locations,
        steepness,
        switch_dim=switch_dim,
    )

    assert_allclose(kernel(X), reference_gram_matrix)
