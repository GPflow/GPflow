# Copyright 2019-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import tensorflow_probability as tfp

from .. import config
from .misc import to_default_float

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


def triangular(positive_diag: bool = False, diag_bijector=None) -> tfp.bijectors.Bijector:
    """
    Returns instance of a triangular bijector.

    :param positive_diag: if True, constrains the diagonal to be positive
    :param diag_bijector: passed through to FillScaleTriL for positive_diag
    """
    if positive_diag:
        return tfp.bijectors.FillScaleTriL(diag_bijector=diag_bijector)
    else:
        assert diag_bijector is None, "should not pass diag_bijector when positive_diag is False"
        return tfp.bijectors.FillTriangular()
