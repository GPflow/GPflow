from typing import Optional, Type

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.config import Config, as_context
from gpflow.utilities import positive, triangular, triangular_size


@pytest.mark.parametrize(
    "env_lower, override_lower",
    [
        (0.1, None),  # ensure default from config is applied
        (0.0, 0.2),  # ensure override is applied
        (0.3, 0.4),  # ensure local overrides config
    ],
)
def test_positive_lower(env_lower: float, override_lower: Optional[float]) -> None:
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
def test_positive_bijector(
    env_bijector: str,
    override_bijector: Optional[str],
    expected_class: Type[tfp.bijectors.Bijector],
) -> None:
    with as_context(Config(positive_bijector=env_bijector, positive_minimum=0.0)):
        bijector = positive(base=override_bijector)
        assert isinstance(bijector, expected_class)


def test_positive_calculation_order() -> None:
    value, lower = -10.0, 10.0
    expected = np.exp(value) + lower
    with as_context(Config(positive_bijector="exp", positive_minimum=lower)):
        result = positive()(value).numpy()
    assert np.isclose(result, expected)
    assert result >= lower


def test_triangular() -> None:
    assert isinstance(triangular(), tfp.bijectors.FillTriangular)


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 6),
        (4, 10),
    ],
)
def test_triangular_size(n: int, expected: int) -> None:
    actual = triangular_size(tf.constant(n))
    assert actual.dtype.is_integer
    assert expected == actual
