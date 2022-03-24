import numpy as np

from .base import AnyNDArray
from .kernels import Kernel


def assert_psd_matrix(A: AnyNDArray, tol: float = 1e-12):
    assert np.linalg.eigvals(A).min() > -tol, "test for positive semi definite matrix"


def test_kernel(kernel: Kernel, X: AnyNDArray, X2: AnyNDArray):
    N, D = X.shape
    N2, D2 = X2.shape
    assert D == D2
    assert N != N2

    kX = kernel(X).numpy()
    assert kX.shape == (N, N)
    kXX = kernel(X, X).numpy()
    assert kXX.shape == (N, N)
    np.testing.assert_allclose(kX, kXX)
    np.testing.assert_allclose(kX, kX.T)
    assert_psd_matrix(kX)

    kXX2 = kernel(X, X2).numpy()
    assert kXX2.shape == (N, N2)

    kX2X = kernel(X2, X).numpy()
    np.testing.assert_allclose(kXX2, kX2X.T)

    kXdiag = kernel(X, full_cov=False).numpy()
    assert kXdiag.shape == (N,)
    np.testing.assert_allclose(np.diag(kX), kXdiag)
