import numpy as np

import gpflow


def test_changepoint_with_X1_X2() -> None:
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
