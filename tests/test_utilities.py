import numpy as np
import pytest
import tensorflow_probability as tfp

from gpflow.config import Config, as_context
from gpflow.utilities import positive, triangular


@pytest.mark.parametrize("env_lower, override_lower", [
    (0.1, None),  # ensure default from config is applied
    (None, 0.2),  # ensure override is applied
    (0.3, 0.4),  # ensure local overrides config
])
def test_positive_lower(env_lower, override_lower):
    expected_lower = override_lower or env_lower
    with as_context(Config(positive_bijector=tfp.bijectors.Softplus(), positive_minimum=env_lower)):
        bijector = positive(lower=override_lower)
        assert isinstance(bijector, tfp.bijectors.Chain)
        assert np.isclose(bijector.bijectors[1].shift, expected_lower)


@pytest.mark.parametrize("env_bijector, override_bijector, expected_class", [
    (tfp.bijectors.Softplus(), None, tfp.bijectors.Softplus),
    (tfp.bijectors.Softplus(), tfp.bijectors.Exp(), tfp.bijectors.Exp),
    (tfp.bijectors.Exp(), None, tfp.bijectors.Exp),
    (tfp.bijectors.Exp(), tfp.bijectors.Softplus(hinge_softness=2.0), tfp.bijectors.Softplus),
])
def test_positive_bijector(env_bijector, override_bijector, expected_class):
    with as_context(Config(positive_bijector=env_bijector, positive_minimum=None)):
        bijector = positive(base=override_bijector)
        assert isinstance(bijector, expected_class)


def test_triangular():
    assert isinstance(triangular(), tfp.bijectors.FillTriangular)
