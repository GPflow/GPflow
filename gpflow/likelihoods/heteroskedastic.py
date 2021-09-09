# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

"""
Likelihoods which require the input location X in addition to F and Y
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import HeteroskedasticTFPConditional
from ..base import Parameter
from ..utilities import positive


class ParameterisedGaussianLikelihood(HeteroskedasticTFPConditional):
    r"""
    The HeteroskedasticGaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with a variance which potentially evolves with input location.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self, **kwargs):
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """

        super().__init__(tfp.distributions.Normal, self._scale_transform, **kwargs)

    def _scale_transform(self, X):
        """ Determine the likelihood variance at the specified input locations X. """
        raise NotImplementedError
