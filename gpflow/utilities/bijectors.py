from typing import Optional

import tensorflow_probability as tfp

from .utilities import to_default_float

__all__ = ["positive", "triangular"]


def positive(lower: Optional[float] = None):
    if lower is None:
        return tfp.bijectors.Softplus()

    shift = to_default_float(lower)
    bijectors = [tfp.bijectors.AffineScalar(shift=shift), tfp.bijectors.Softplus()]
    return tfp.bijectors.Chain(bijectors)


triangular = tfp.bijectors.FillTriangular
