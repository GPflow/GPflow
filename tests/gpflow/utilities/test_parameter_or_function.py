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

import numpy as np
import pytest
import tensorflow_probability as tfp
from numpy.testing import assert_allclose

from gpflow.base import Parameter
from gpflow.functions import Linear
from gpflow.utilities.parameter_or_function import (
    evaluate_parameter_or_function,
    prepare_parameter_or_function,
)


def test_prepare_parameter_or_function__constant__no_bound() -> None:
    initial = 5.0
    param = prepare_parameter_or_function(initial)
    assert isinstance(param, Parameter)
    assert isinstance(param.transform, tfp.bijectors.Identity)
    assert_allclose(initial, param)

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    assert_allclose(initial, evaluate_parameter_or_function(param, X))


def test_prepare_parameter_or_function__constant__bound() -> None:
    initial = 5.0

    with pytest.raises(Exception):
        prepare_parameter_or_function(initial, lower_bound=initial + 1e-3)

    lower_bound = initial - 1e-3
    param = prepare_parameter_or_function(initial, lower_bound=lower_bound)
    assert isinstance(param, Parameter)
    assert isinstance(param.transform, tfp.bijectors.Chain)
    assert_allclose(initial, param)

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    assert_allclose(initial, evaluate_parameter_or_function(param, X, lower_bound=lower_bound))


def test_prepare_parameter_or_function__function__no_bound() -> None:
    initial = Linear([[0.5], [2.0]], 1.0)
    func = prepare_parameter_or_function(initial)
    assert initial is func

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    assert_allclose(
        [[1.0], [3.0], [5.0], [1.5], [3.5], [5.5]], evaluate_parameter_or_function(func, X)
    )


def test_prepare_parameter_or_function__function__bound() -> None:
    initial = Linear([[0.5], [2.0]], 1.0)
    lower_bound = 3.2
    func = prepare_parameter_or_function(initial, lower_bound=lower_bound)
    assert initial is func

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    assert_allclose(
        [[3.2], [3.2], [5.0], [3.2], [3.5], [5.5]],
        evaluate_parameter_or_function(func, X, lower_bound=lower_bound),
    )
