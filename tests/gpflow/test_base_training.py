# Copyright 2018 the GPflow authors.
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


from typing import Any, Mapping

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.models import SVGP
from gpflow.utilities import leaf_components, multiple_assign, read_values

rng = np.random.RandomState(0)


class Data:
    H0 = 5
    H1 = 2
    M = 10
    D = 1
    Z = rng.rand(M, 1)
    ls = 2.0
    ls_new = 1.5
    var = 1.0
    var_new = 10.0


# ------------------------------------------
# Fixtures
# ------------------------------------------


@pytest.fixture
def model() -> SVGP:
    return SVGP(
        kernel=gpflow.kernels.SquaredExponential(lengthscales=Data.ls, variance=Data.var),
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=Data.Z,
        q_diag=True,
    )


# ------------------------------------------
# Reference
# ------------------------------------------

model_param_updates = {
    ".kernel.lengthscales": Data.ls_new,
    ".likelihood.variance": Data.var_new,
    ".inducing_variable.Z": np.zeros_like(Data.Z),
    ".q_sqrt": 0.5 * np.ones((Data.M, 1)),
}
model_wrong_path = [
    {"kernel.lengthscales": Data.ls_new},
    {".Gaussian.variance": Data.var_new},
    {"inducing_variable.Z": np.zeros_like(Data.Z)},
    {".q_std": 0.5 * np.ones((Data.M, 1))},
]

model_wrong_value = [
    {".likelihood.variance": np.ones((2, 1), dtype=np.int32)},
    {".inducing_variable.Z": [1, 2, 3]},
]


@pytest.mark.parametrize("var_update_dict", [model_param_updates])
def test_multiple_assign_updates_correct_values(
    model: SVGP, var_update_dict: Mapping[str, Any]
) -> None:
    old_value_dict = leaf_components(model)
    multiple_assign(model, var_update_dict)
    for path, variable in leaf_components(model).items():
        if path in var_update_dict.keys():
            np.testing.assert_almost_equal(variable.numpy(), var_update_dict[path], decimal=7)
        else:
            np.testing.assert_equal(variable.numpy(), old_value_dict[path].numpy())


@pytest.mark.parametrize("wrong_var_update_dict", model_wrong_path)
def test_multiple_assign_fails_with_invalid_path(
    model: SVGP, wrong_var_update_dict: Mapping[str, Any]
) -> None:
    with pytest.raises(KeyError):
        multiple_assign(model, wrong_var_update_dict)


@pytest.mark.parametrize("wrong_var_update_dict", model_wrong_value)
def test_multiple_assign_fails_with_invalid_values(
    model: SVGP, wrong_var_update_dict: Mapping[str, Any]
) -> None:
    with pytest.raises(ValueError):
        multiple_assign(model, wrong_var_update_dict)


def test_dict_utilities(model: SVGP) -> None:
    """
    Test both `parameter_dict()` and `read_values()`
    """

    class SubModule(tf.Module):
        def __init__(self) -> None:
            self.parameter = gpflow.Parameter(1.0)
            self.variable = tf.Variable(1.0)

    class Module(tf.Module):
        def __init__(self) -> None:
            self.submodule = SubModule()
            self.top_parameter = gpflow.Parameter(3.0)

    m = Module()
    params = gpflow.utilities.parameter_dict(m)
    # {
    #   ".submodule.parameter": <parameter object>,
    #   ".submodule.variable": <variable object>
    # }
    assert list(params.keys()) == [
        ".submodule.parameter",
        ".submodule.variable",
        ".top_parameter",
    ]
    assert list(params.values()) == [
        m.submodule.parameter,
        m.submodule.variable,
        m.top_parameter,
    ]

    for k, v in read_values(m).items():
        assert params[k].numpy() == v
