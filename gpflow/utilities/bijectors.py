from typing import Optional

import tensorflow_probability as tfp

from .. import config
from .utilities import to_default_float


__all__ = ["positive", "triangular"]


_POSITIVE_BIJECTOR_MAP = {
    "exp": tfp.bijectors.Exp,
    "softplus": tfp.bijectors.Softplus,
}


class Shift(tfp.bijectors.AffineScalar):
    """Simple subclass so printed name is cleaner."""

    def __init__(self, shift=None, validate_args=False, name='shift'):
        super().__init__(shift=shift, validate_args=validate_args, name=name)


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
        shift = Shift(shift=to_default_float(lower))
        bijector = tfp.bijectors.Chain([bijector, shift])
    return bijector


def _get_base_positive_bijector(bijector_name: Optional[str] = None):
    if bijector_name is None:
        bijector_name = config.default_positive_bijector()
    return _POSITIVE_BIJECTOR_MAP[bijector_name]()


def triangular():
    """
    Returns instance of a triangular bijector.
    """
    return tfp.bijectors.FillTriangular()
