import numpy as np
import pytest
import tensorflow_probability as tfp

import gpflow
from gpflow.config import Config, as_context
from gpflow.utilities import positive, triangular


@pytest.mark.parametrize(
    "env_lower, override_lower",
    [
        (0.1, None),  # ensure default from config is applied
        (None, 0.2),  # ensure override is applied
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
    with as_context(Config(positive_bijector=env_bijector, positive_minimum=None)):
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
