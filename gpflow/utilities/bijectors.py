from typing import Optional

import tensorflow_probability as tfp

from .. import config
from .utilities import to_default_float

__all__ = ["positive", "triangular"]


def positive(lower: Optional[float] = None):
    lower_value = config.default_positive_minimum()
    if lower_value is None:
        return tfp.bijectors.Softplus()

    shift = to_default_float(lower_value)
    bijectors = [tfp.bijectors.AffineScalar(shift=shift), tfp.bijectors.Softplus()]
    return tfp.bijectors.Chain(bijectors)


triangular = tfp.bijectors.FillTriangular
