import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.base import TensorType
from gpflow.utilities import add_likelihood_noise_cov, add_noise_cov, assert_params_false


def test_assert_params_false__False() -> None:
    assert_params_false(test_assert_params_false__False, foo=False, bar=False)


def test_assert_params_false__True() -> None:
    with pytest.raises(NotImplementedError):
        assert_params_false(test_assert_params_false__True, foo=False, bar=True)


@pytest.mark.parametrize(
    "input_tensor,variance,expected_tensor",
    [
        (
            tf.constant([[1.0, 0.5], [0.5, 1.0]]),
            gpflow.Parameter(1.0, dtype=tf.float32),
            tf.constant([[2.0, 0.5], [0.5, 2.0]]),
        ),
        (
            tf.constant(
                [
                    [
                        [
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                    ],
                    [
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                        ],
                    ],
                ]
            ),
            gpflow.Parameter(0.5, dtype=tf.float32),
            tf.constant(
                [
                    [
                        [
                            [0.5, 0.0],
                            [0.0, 0.5],
                        ],
                    ],
                    [
                        [
                            [1.5, 1.0],
                            [1.0, 1.5],
                        ],
                    ],
                ]
            ),
        ),
        (
            tf.constant([[1.0, 0.5], [0.5, 1.0]]),
            gpflow.Parameter([1.0, 2.0], dtype=tf.float32),
            tf.constant([[2.0, 0.5], [0.5, 3.0]]),
        ),
    ],
)
def test_add_noise_cov(
    input_tensor: TensorType, variance: TensorType, expected_tensor: TensorType
) -> None:
    actual_tensor = add_noise_cov(input_tensor, variance)
    np.testing.assert_allclose(actual_tensor, expected_tensor)


def test_add_likelihood_noise_cov() -> None:
    K = tf.constant(
        [
            [0.11, 0.12, 0.13],
            [0.21, 0.22, 0.23],
            [0.31, 0.32, 0.33],
        ],
        dtype=gpflow.default_float(),
    )
    variance = gpflow.functions.Linear(A=[[0.2]], b=0.1)
    likelihood = gpflow.likelihoods.Gaussian(variance=variance)
    X = tf.constant([[2.0], [4.0], [3.0]], dtype=gpflow.default_float())
    actual_tensor = add_likelihood_noise_cov(K, likelihood, X)

    np.testing.assert_allclose(
        [
            [0.61, 0.12, 0.13],
            [0.21, 1.12, 0.23],
            [0.31, 0.32, 1.03],
        ],
        actual_tensor,
    )
