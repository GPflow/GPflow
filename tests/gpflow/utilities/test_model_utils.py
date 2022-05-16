import pytest
import tensorflow as tf

import gpflow
from gpflow.base import TensorType
from gpflow.utilities import add_noise_cov


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
    ],
)
def test_add_noise_cov(
    input_tensor: TensorType, variance: TensorType, expected_tensor: TensorType
) -> None:
    actual_tensor = add_noise_cov(input_tensor, variance)
    tf.debugging.assert_equal(actual_tensor, expected_tensor)
