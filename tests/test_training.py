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


import numpy as np
import pytest

import gpflow
from gpflow.utilities.printing import leaf_components
from gpflow.utilities.training import multiple_assign

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
    var_new = 10.


# ------------------------------------------
# Helpers
# ------------------------------------------


kernel = gpflow.kernels.RBF(lengthscale=Data.ls, variance=Data.var)
kernel_param_updates = {'RBF.lengthscale': Data.ls_new}

model = gpflow.models.SVGP(
    kernel=kernel,
    likelihood=gpflow.likelihoods.Gaussian(),
    feature=Data.Z,
    q_diag=True
)
model_param_updates = {
    'SVGP.kernel.lengthscale': Data.ls_new,
    'SVGP.likelihood.variance': Data.var_new,
    'SVGP.feature.Z': np.zeros_like(Data.Z),
    'SVGP.q_sqrt': 0.5 * np.ones((Data.M, 1))
}
model_wrong_path = [
    {'kernel.lengthscale': Data.ls_new},
    {'SVGP.Gaussian.variance': Data.var_new},
    {'feature.Z': np.zeros_like(Data.Z)},
    {'SVGP.q_std': 0.5 * np.ones((Data.M, 1))}
]

model_wrong_value = [
    {'SVGP.likelihood.variance': np.ones((2, 1), dtype=np.int32)},
    {'SVGP.feature.Z': [1, 2, 3]}
]


@pytest.mark.parametrize('module, var_update_dict', [
    (kernel, kernel_param_updates),
    (model, model_param_updates)
])
def test_multiple_assign_updates_correct_values(module, var_update_dict):
    old_value_dict = leaf_components(module).copy()
    multiple_assign(module, var_update_dict)
    for path, variable in leaf_components(module).items():
        if path in var_update_dict.keys():
            np.testing.assert_almost_equal(variable.value().numpy(), var_update_dict[path], decimal=7)
        else:
            np.testing.assert_equal(variable.value().numpy(), old_value_dict[path].value().numpy())


@pytest.mark.parametrize('module', [model])
@pytest.mark.parametrize('wrong_var_update_dict', model_wrong_path)
def test_multiple_assign_fails_with_invalid_path(module, wrong_var_update_dict):
    with pytest.raises(KeyError):
        multiple_assign(module, wrong_var_update_dict)


@pytest.mark.parametrize('module', [model])
@pytest.mark.parametrize('wrong_var_update_dict', model_wrong_value)
def test_multiple_assign_fails_with_invalid_values(module, wrong_var_update_dict):
    with pytest.raises(ValueError):
        multiple_assign(module, wrong_var_update_dict)
