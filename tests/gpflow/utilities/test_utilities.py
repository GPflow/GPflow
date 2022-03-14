from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.base import AnyNDArray
from gpflow.models.util import data_input_to_tensor


def test_select_parameters_with_prior() -> None:
    kernel = gpflow.kernels.SquaredExponential()
    params = gpflow.utilities.select_dict_parameters_with_prior(kernel)
    assert params == {}

    kernel.variance.prior = tfp.distributions.Gamma(1.0, 1.0)
    params = gpflow.utilities.select_dict_parameters_with_prior(kernel)
    assert len(params) == 1


def test_data_input_to_tensor() -> None:
    input1 = (1.0, (2.0,))
    output1 = data_input_to_tensor(input1)
    assert output1[0].dtype == tf.float64
    assert output1[1][0].dtype == tf.float64

    input2 = (1.0, [2.0])
    output2 = data_input_to_tensor(input2)
    assert output2[0].dtype == tf.float64
    assert output2[1][0].dtype == tf.float64

    input3: Tuple[float, Tuple[AnyNDArray, AnyNDArray]] = (
        1.0,
        (np.arange(3, dtype=np.float16),) * 2,
    )
    output3 = data_input_to_tensor(input3)
    assert output3[0].dtype == tf.float64
    assert output3[1][0].dtype == tf.float16
    assert output3[1][1].dtype == tf.float16
