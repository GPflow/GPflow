# Copyright 2020 the GPflow authors.
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


import gpflow
import pytest
import tensorflow as tf
from gpflow.utilities import positive


def test_parameter_assign_validation():
    with pytest.raises(tf.errors.InvalidArgumentError):
        param = gpflow.Parameter(0.0, transform=positive())

    param = gpflow.Parameter(0.1, transform=positive())
    param.assign(0.2)
    with pytest.raises(tf.errors.InvalidArgumentError):
        param.assign(0.0)
