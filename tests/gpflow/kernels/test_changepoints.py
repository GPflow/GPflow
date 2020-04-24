import numpy as np
import gpflow
import numpy as np
import gpflow


def test_changepoint_with_X1_X2():
    X = np.linspace(0, 100, 100).reshape(100, 1)
    base_k1 = gpflow.kernels.Matern32(lengthscales=0.2)
    base_k2 = gpflow.kernels.Matern32(lengthscales=2.0)
    k = gpflow.kernels.ChangePoints([base_k1, base_k2], [0.0], steepness=5.0)
    K = k(X)
    assert K.shape == [100, 100]

    N = 25
    X2 = np.linspace(0, 50, N).reshape(N, 1)
    K = k(X, X2)
    assert K.shape == [100, 25]


if __name__ == "__main__":
    test_changepoint_with_X1_X2()
