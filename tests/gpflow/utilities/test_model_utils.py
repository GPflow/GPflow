import pytest
import tensorflow as tf

import gpflow
from gpflow.utilities import add_noise_cov


@pytest.mark.parametrize("input_tensor", [tf.constant([[1.0, 0.5], [0.5, 1.0]])])
@pytest.mark.parametrize("variance", [gpflow.Parameter(1.0, dtype=tf.float32)])
@pytest.mark.parametrize("expected_tensor", [tf.constant([[2.0, 0.5], [0.5, 2.0]])])
def test_add_noise_cov(input_tensor, variance, expected_tensor):
    actual_tensor = add_noise_cov(input_tensor, variance)
    tf.debugging.assert_equal(actual_tensor, expected_tensor)
