# Copyright 2017 the GPflow authors.
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

import tensorflow as tf

import numpy as np
from numpy.testing import assert_array_equal

import gpflow
from gpflow.test_util import GPflowTestCase


class TestTensorConverter(GPflowTestCase):
    def test_failures(self):
        values = ['', 'test', 1., None, object()]
        for v in values:
            with self.assertRaises(ValueError, msg='Raises at "{}"'.format(v)):
                gpflow.core.tensor_converter.TensorConverter.tensor_mode(v)

        p = gpflow.core.parentable.Parentable()
        p._parent = p
        with self.assertRaises(gpflow.GPflowError):
            gpflow.core.tensor_converter.TensorConverter.tensor_mode(p)
