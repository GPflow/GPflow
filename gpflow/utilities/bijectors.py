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
    if lower is None:
        lower = config.default_positive_minimum()
    if lower is None:
        return _get_base_positive_bijector(bijector)

    shift = to_default_float(lower)
    return tfp.bijectors.Chain([
        tfp.bijectors.AffineScalar(shift=shift),
        _get_base_positive_bijector(bijector),
    ])


def _get_base_positive_bijector(bijector_name: Optional[str] = None):
    if bijector_name is None:
        bijector_name = config.default_positive_bijector()
    return _POSITIVE_BIJECTOR_MAP[bijector_name]()


triangular = tfp.bijectors.FillTriangular
