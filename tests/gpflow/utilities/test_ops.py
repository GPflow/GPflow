from typing import Union, cast

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shapes
from gpflow.utilities.ops import difference_matrix


@check_shapes(
    "X: [N, D]",
    "Q: []",
    "return: [N, Q]",
)
def pca_reduce(X: Union[AnyNDArray, tf.Tensor], Q: int) -> AnyNDArray:
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    if Q > X.shape[1]:  # pragma: no cover
        raise ValueError("Cannot have more latent dimensions than observed")
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    assert isinstance(X, np.ndarray)  # Hint for numpy
    # TODO why not use tf.linalg.eigh?
    evals, evecs = np.linalg.eigh(np.cov(X.T))
    W = evecs[:, -Q:]
    return cast(AnyNDArray, (X - X.mean(0)).dot(W))


@pytest.mark.parametrize("N", [3, 7])
@pytest.mark.parametrize("D", [2, 5, 9])
@pytest.mark.parametrize("Q", [2, 5, 9])
def test_pca_reduce_numpy_equivalence(N: int, D: int, Q: int) -> None:
    X = np.random.randn(N, D)

    if Q > D:
        with pytest.raises(ValueError):
            gpflow.utilities.ops.pca_reduce(tf.convert_to_tensor(X), Q)

    else:
        np_result = pca_reduce(X, Q)
        tf_result = gpflow.utilities.ops.pca_reduce(tf.convert_to_tensor(X), Q).numpy()
        assert np_result.shape == tf_result.shape == (N, Q)

        for i in range(Q):
            # PCA does not necessarily preserve the overall sign, so also accept it to flip
            tf_column = tf_result[:, i]
            np_column = np_result[:, i]
            assert np.allclose(tf_column, np_column) or np.allclose(tf_column, -np_column)


def test_difference_matrix_broadcasting_symmetric() -> None:
    X = np.random.randn(5, 4, 3, 2)
    d = difference_matrix(X, None)
    assert d.shape == (5, 4, 3, 3, 2)


def test_difference_matrix_broadcasting_cross() -> None:
    X = np.random.randn(2, 3, 4, 5)
    X2 = np.random.randn(8, 7, 6, 5)
    d = difference_matrix(X, X2)
    assert d.shape == (2, 3, 4, 8, 7, 6, 5)
