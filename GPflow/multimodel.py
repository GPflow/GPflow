# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
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


from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .param import Param
from .svgp import SVGP
from . import transforms, conditionals, kullback_leiblers
from .mean_functions import Zero
from .tf_wraps import eye
from ._settings import settings
from .minibatch import MinibatchData

class ChainedGP(SVGP):
    def __init__(self, X, Y, kern, likelihood, Z, mean_function=Zero(),
                 q_diag=False, whiten=True, minibatch_size=None):
        assert isinstance(likelihood, ChainedLikelihood)
        SVGP.__init__(self, X, Y, kern, likelihood, mean_function,
                      num_latent=2, q_diag=False,
                      whiten=True, minibatch_size=None)
