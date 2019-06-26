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
import tensorflow as tf

import gpflow
from gpflow.utilities.printing import leaf_components, _merge_leaf_components

import pytest

rng = np.random.RandomState(0)


class Data:
    H0 = 5
    H1 = 2
    M = 10
    D = 1
    Z = rng.rand(M, 1)
    ls = 2.0
    var = 1.0


# ------------------------------------------
# Helpers
# ------------------------------------------


class A(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.var_trainable = tf.Variable(tf.zeros((2, 2, 1)), trainable=True)
        self.var_fixed = tf.Variable(tf.ones((2, 2, 1)), trainable=False)


class B(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
        self.submodule_list = [A(), A()]
        self.var_trainable = tf.Variable(tf.zeros((2, 2, 1)), trainable=True)
        self.var_fixed = tf.Variable(tf.ones((2, 2, 1)), trainable=False)


example_tf_module = A()
example_module_list = B()
kernel = gpflow.kernels.RBF(lengthscale=Data.ls, variance=Data.var)
kernel.lengthscale.trainable = False
model_gp = gpflow.models.SVGP(kernel=kernel, likelihood=gpflow.likelihoods.Gaussian(),
                              feature=Data.Z, q_diag=True)
model_gp.q_mu.trainable = False


@pytest.fixture
def example_dag_module():
    model_dag = model_gp
    model_dag.kernel.variance = model_dag.kernel.lengthscale
    return model_dag


example_tf_module_variable_dict = {
    'A.var_trainable': {
        'value': np.zeros((2, 2, 1)),
        'trainable': True,
        'shape': (2, 2, 1)
    },
    'A.var_fixed': {
        'value': np.ones((2, 2, 1)),
        'trainable': False,
        'shape': (2, 2, 1)
    },
}

example_module_list_variable_dict = {
    'A_0.var_trainable': example_tf_module_variable_dict['A.var_trainable'],
    'A_0.var_fixed': example_tf_module_variable_dict['A.var_fixed'],
    'A_1.var_trainable': example_tf_module_variable_dict['A.var_trainable'],
    'A_1.var_fixed': example_tf_module_variable_dict['A.var_fixed'],
    'B.var_trainable': example_tf_module_variable_dict['A.var_trainable'],
    'B.var_fixed': example_tf_module_variable_dict['A.var_fixed'],
}

kernel_param_dict = {
    'RBF.lengthscale': {
        'value': Data.ls,
        'trainable': False,
        'shape': ()
    },
    'RBF.variance': {
        'value': Data.var,
        'trainable': True,
        'shape': ()
    }
}

model_gp_param_dict = {
    'kernel.lengthscale': kernel_param_dict['RBF.lengthscale'],
    'kernel.variance': kernel_param_dict['RBF.variance'],
    'likelihood.variance': {
        'value': 1.0,
        'trainable': True,
        'shape': ()
    },
    'feature.Z': {
        'value': Data.Z,
        'trainable': True,
        'shape': (Data.M, Data.D)
    },
    'SVGP.q_mu': {
        'value': np.zeros((Data.M, 1)),
        'trainable': False,
        'shape': (Data.M, 1)
    },
    'SVGP.q_sqrt': {
        'value': np.ones((Data.M, 1)),
        'trainable': True,
        'shape': (Data.M, 1)
    }
}

example_dag_module_param_dict = {
    'SVGP.kernel.variance\nSVGP.kernel.lengthscale': kernel_param_dict['RBF.lengthscale'],
    'SVGP.likelihood.variance': {
        'value': 1.0,
        'trainable': True,
        'shape': ()
    },
    'SVGP.feature.Z': {
        'value': Data.Z,
        'trainable': True,
        'shape': (Data.M, Data.D)
    },
    'SVGP.q_mu': {
        'value': np.zeros((Data.M, 1)),
        'trainable': False,
        'shape': (Data.M, 1)
    },
    'SVGP.q_sqrt': {
        'value': np.ones((Data.M, 1)),
        'trainable': True,
        'shape': (Data.M, 1)
    }
}


@pytest.mark.parametrize('module', [example_tf_module, example_module_list, kernel, model_gp])
def test_leaf_components_only_returns_parameters_and_variables(module):
    for path, variable in leaf_components(module).items():
        assert isinstance(variable, tf.Variable) or isinstance(variable, gpflow.Parameter)


@pytest.mark.parametrize('module, expected_param_dicts', [(kernel, kernel_param_dict),
                                                          (model_gp, model_gp_param_dict)])
def test_leaf_components_registers_variable_properties(module, expected_param_dicts):
    for path, variable in leaf_components(module).items():
        param_name = path.split('.')[-2] + '.' + path.split('.')[-1]
        assert isinstance(variable, gpflow.Parameter)
        np.testing.assert_equal(variable.value().numpy(), expected_param_dicts[param_name]['value'])
        assert variable.trainable == expected_param_dicts[param_name]['trainable']
        assert variable.shape == expected_param_dicts[param_name]['shape']


@pytest.mark.parametrize('module, expected_var_dicts', [
    (example_tf_module, example_tf_module_variable_dict),
    (example_module_list, example_module_list_variable_dict),
])
def test_leaf_components_registers_param_properties(module, expected_var_dicts):
    for path, variable in leaf_components(module).items():
        var_name = path.split('.')[-2] + '.' + path.split('.')[-1]
        assert isinstance(variable, tf.Variable)
        np.testing.assert_equal(variable.numpy(), expected_var_dicts[var_name]['value'])
        assert variable.trainable == expected_var_dicts[var_name]['trainable']
        assert variable.shape == expected_var_dicts[var_name]['shape']


@pytest.mark.parametrize('expected_var_dicts', [example_dag_module_param_dict])
def test_merge_leaf_components_merges_keys_with_same_values(example_dag_module, expected_var_dicts):
    leaf_components_dict = leaf_components(example_dag_module)
    for path, variable in _merge_leaf_components(leaf_components_dict).items():
        assert path in expected_var_dicts
        for sub_path in path.split('\n'):
            assert sub_path in leaf_components_dict
            assert leaf_components_dict[sub_path] is variable
