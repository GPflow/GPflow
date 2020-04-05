from typing import Optional

import tensorflow_probability as tfp

from .. import config
from .utilities import to_default_float


__all__ = ["positive", "triangular"]


def positive(lower: Optional[float] = None, base: Optional[str] = None) -> tfp.bijectors.Bijector:
    """
    Returns a positive bijector (a reversible transformation from real to positive numbers).

    :param lower: overrides default lower bound
        (if None, defaults to gpflow.config.default_positive_minimum())
    :param base: overrides base positive bijector
        (if None, defaults to gpflow.config.default_positive_bijector())
    :returns: a bijector instance
    """
    bijector = base if base is not None else config.default_positive_bijector()
    bijector = config.positive_bijector_type_map()[bijector.lower()]()

    lower_bound = lower if lower is not None else config.default_positive_minimum()

    if lower_bound != 0.0:
        shift = tfp.bijectors.Shift(to_default_float(lower_bound))
        bijector = tfp.bijectors.Chain([shift, bijector])  # from unconstrained to constrained
    return bijector


def triangular() -> tfp.bijectors.Bijector:
    """
    Returns instance of a triangular bijector.
    """
    return tfp.bijectors.FillTriangular()
