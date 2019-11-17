from typing import Optional

import tensorflow_probability as tfp

from .. import config
from .utilities import to_default_float


__all__ = ["positive", "triangular"]


_POSITIVE_BIJECTOR_MAP = {
    "exp": tfp.bijectors.Exp,
    "softplus": tfp.bijectors.Softplus,
}


def positive(lower: Optional[float] = None,
             bijector: Optional[str] = None) -> tfp.bijectors.Bijector:
    """
    Returns a positive bijector (a reversible transformation from real to positive numbers).

    :param lower: lower bound override (defaults to config.default_positive_minimum())
    :param bijector: bijector method override (defaults to config.default_positive_bijector())
    :returns: a bijector instance
    """
    bijector = _get_base_positive_bijector(bijector)
    if lower is None:
        lower = config.default_positive_minimum()
    if lower is not None:
        # Apply lower bound shift after applying base positive bijector
        shift = tfp.bijectors.AffineScalar(shift=to_default_float(lower))
        bijector = tfp.bijectors.Chain([bijector, shift])
    return bijector


def _get_base_positive_bijector(name: Optional[str] = None) -> tfp.bijectors.Bijector:
    if name is None:
        name = config.default_positive_bijector()
    return _POSITIVE_BIJECTOR_MAP[name]()


def triangular() -> tfp.bijectors.Bijector:
    """
    Returns instance of a triangular bijector.
    """
    return tfp.bijectors.FillTriangular()
