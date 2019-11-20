from typing import Optional

import tensorflow_probability as tfp

from .. import config
from .utilities import to_default_float


__all__ = ["positive", "triangular"]


def positive(lower: Optional[float] = None,
             base: Optional[tfp.bijectors.Bijector] = None) -> tfp.bijectors.Bijector:
    """
    Returns a positive bijector (a reversible transformation from real to positive numbers).

    :param lower: lower bound override (defaults to config.default_positive_minimum())
    :param base: overrides the default base positive bijector (defaults to config.default_positive_bijector())
    :returns: a bijector instance
    """
    bijector = base if base is not None else config.default_positive_bijector()
    if lower is None:
        lower = config.default_positive_minimum()
    if lower is not None:
        # Apply lower bound shift after applying base positive bijector
        shift = tfp.bijectors.AffineScalar(shift=to_default_float(lower))
        bijector = tfp.bijectors.Chain([bijector, shift])
    return bijector


def triangular() -> tfp.bijectors.Bijector:
    """
    Returns instance of a triangular bijector.
    """
    return tfp.bijectors.FillTriangular()
