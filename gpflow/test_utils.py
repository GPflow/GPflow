import numpy as np

from .kernels import Kernel


def test_kernel(kernel: Kernel, X: np.ndarray, X2: np.ndarray):
    N, D = X.shape
    N2, D2 = X2.shape
    assert D == D2
    assert N1 != N2

    kX = kernel(X).numpy()
    assert kX.shape == (N, N)
    kXX = kernel(X, X).numpy()
    assert kXX.shape == (N, N)
    np.testing.assert_allclose(kX, kXX)
    np.testing.assert_allclose(kX, kX.T)
    assert np.linalg.eigvals(kX).min() > -1e-6, "kernel matrix should be positive definite"

    kXX2 = kernel(X, X2).numpy()
    assert kXX2.shape == (N, N2)

    kX2X = kernel(X2, X).numpy()
    np.testing.assert_allclose(kXX2, kX2X.T)

    kXdiag = kernel(X, full_cov=False).numpy()
    assert kXdiag.shape == (N,)
    np.testing.assert_allclose(np.diag(kX), kXdiag)
