# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews, Alexis Boukouvalas
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
from . import densities, transforms
import tensorflow as tf
import numpy as np
from .param import Parameterized, Param, ParamList
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class ChainedLikelihood(Parameterized):
    def __init__(self):
        Parameterized.__init__(self)
        self.scoped_keys.extend(['logp'])
        self.num_gauss_hermite_points = 20

    def logp(self, F, Y):
        """
        Return the log density of the data given the function values.
        """
        raise NotImplementedError("implement the logp function\
                                  for this likelihood")
