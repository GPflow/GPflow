import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.config import Config, as_context
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import positive, triangular
from gpflow.utilities.ops import difference_matrix


@pytest.mark.parametrize(
    "env_lower, override_lower",
    [
        (0.1, None),  # ensure default from config is applied
        (0.0, 0.2),  # ensure override is applied
        (0.3, 0.4),  # ensure local overrides config
    ],
)
def test_positive_lower(env_lower, override_lower):
    expected_lower = override_lower or env_lower
    with as_context(Config(positive_bijector="softplus", positive_minimum=env_lower)):
        bijector = positive(lower=override_lower)
        assert isinstance(bijector, tfp.bijectors.Chain)
        assert np.isclose(bijector.bijectors[0].shift, expected_lower)


@pytest.mark.parametrize(
    "env_bijector, override_bijector, expected_class",
    [
        ("softplus", None, tfp.bijectors.Softplus),
        ("softplus", "Exp", tfp.bijectors.Exp),
        ("exp", None, tfp.bijectors.Exp),
        ("exp", "Softplus", tfp.bijectors.Softplus),
    ],
)
def test_positive_bijector(env_bijector, override_bijector, expected_class):
    with as_context(Config(positive_bijector=env_bijector, positive_minimum=0.0)):
        bijector = positive(base=override_bijector)
        assert isinstance(bijector, expected_class)


def test_positive_calculation_order():
    value, lower = -10.0, 10.0
    expected = np.exp(value) + lower
    with as_context(Config(positive_bijector="exp", positive_minimum=lower)):
        result = positive()(value).numpy()
    assert np.isclose(result, expected)
    assert result >= lower


def test_triangular():
    assert isinstance(triangular(), tfp.bijectors.FillTriangular)


def test_select_parameters_with_prior():
    kernel = gpflow.kernels.SquaredExponential()
    params = gpflow.utilities.select_dict_parameters_with_prior(kernel)
    assert params == {}

    kernel.variance.prior = tfp.distributions.Gamma(1.0, 1.0)
    params = gpflow.utilities.select_dict_parameters_with_prior(kernel)
    assert len(params) == 1


def test_difference_matrix_broadcasting_symmetric():
    X = np.random.randn(5, 4, 3, 2)
    d = difference_matrix(X, None)
    assert d.shape == (5, 4, 3, 3, 2)


def test_difference_matrix_broadcasting_cross():
    X = np.random.randn(2, 3, 4, 5)
    X2 = np.random.randn(8, 7, 6, 5)
    d = difference_matrix(X, X2)
    assert d.shape == (2, 3, 4, 8, 7, 6, 5)


def test_data_input_to_tensor():
    input1 = (1.0, (2.0,))
    output1 = data_input_to_tensor(input1)
    assert output1[0].dtype == tf.float64
    assert output1[1][0].dtype == tf.float64

    input2 = (1.0, [2.0])
    output2 = data_input_to_tensor(input2)
    assert output2[0].dtype == tf.float64
    assert output2[1][0].dtype == tf.float64

    input3 = (1.0, (np.arange(3, dtype=np.float16),) * 2)
    output3 = data_input_to_tensor(input3)
    assert output3[0].dtype == tf.float64
    assert output3[1][0].dtype == tf.float16
    assert output3[1][1].dtype == tf.float16
