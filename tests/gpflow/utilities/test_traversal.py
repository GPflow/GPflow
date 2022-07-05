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

from typing import Any, Callable, Mapping, Optional, Type

import numpy as np
import pytest
import tensorflow as tf
from _pytest.fixtures import SubRequest
from packaging.version import Version

import gpflow
from gpflow.base import AnyNDArray
from gpflow.config import Config, as_context
from gpflow.utilities import set_trainable
from gpflow.utilities.traversal import (
    _merge_leaf_components,
    leaf_components,
    tabulate_module_summary,
)

rng = np.random.RandomState(0)


class Data:
    H0 = 5
    H1 = 2
    M = 10
    D = 1
    Z: AnyNDArray = 0.5 * np.ones((M, 1))
    ls = 2.0
    var = 1.0


# ------------------------------------------
# Helpers
# ------------------------------------------


class A(tf.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.var_trainable = tf.Variable(tf.zeros((2, 2, 1)), trainable=True)
        self.var_fixed = tf.Variable(tf.ones((2, 2, 1)), trainable=False)


class B(tf.Module):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.submodule_list = [A(), A()]
        self.submodule_dict = dict(a=A(), b=A())
        self.var_trainable = tf.Variable(tf.zeros((2, 2, 1)), trainable=True)
        self.var_fixed = tf.Variable(tf.ones((2, 2, 1)), trainable=False)


class C(tf.keras.Model):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.variable = tf.Variable(tf.zeros((2, 2, 1)), trainable=True)
        self.param = gpflow.Parameter(0.0)
        self.dense = tf.keras.layers.Dense(5)


def create_kernel() -> gpflow.kernels.Kernel:
    kern = gpflow.kernels.SquaredExponential(lengthscales=Data.ls, variance=Data.var)
    set_trainable(kern.lengthscales, False)
    return kern


def create_compose_kernel() -> gpflow.kernels.Kernel:
    kernel = gpflow.kernels.Product(
        [
            gpflow.kernels.Sum([create_kernel(), create_kernel()]),
            gpflow.kernels.Sum([create_kernel(), create_kernel()]),
        ]
    )
    return kernel


def create_model() -> gpflow.models.GPModel:
    kernel = create_kernel()
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance_lower_bound=0.0),
        inducing_variable=Data.Z,
        q_diag=True,
    )
    set_trainable(model.q_mu, False)
    return model


# ------------------------------------------
# Reference
# ------------------------------------------

example_tf_module_variable_dict = {
    "A.var_trainable": {
        "value": np.zeros((2, 2, 1)),
        "trainable": True,
        "shape": (2, 2, 1),
    },
    "A.var_fixed": {
        "value": np.ones((2, 2, 1)),
        "trainable": False,
        "shape": (2, 2, 1),
    },
}

example_module_list_variable_dict = {
    "submodule_list[0].var_trainable": example_tf_module_variable_dict["A.var_trainable"],
    "submodule_list[0].var_fixed": example_tf_module_variable_dict["A.var_fixed"],
    "submodule_list[1].var_trainable": example_tf_module_variable_dict["A.var_trainable"],
    "submodule_list[1].var_fixed": example_tf_module_variable_dict["A.var_fixed"],
    "submodule_dict['a'].var_trainable": example_tf_module_variable_dict["A.var_trainable"],
    "submodule_dict['a'].var_fixed": example_tf_module_variable_dict["A.var_fixed"],
    "submodule_dict['b'].var_trainable": example_tf_module_variable_dict["A.var_trainable"],
    "submodule_dict['b'].var_fixed": example_tf_module_variable_dict["A.var_fixed"],
    "B.var_trainable": example_tf_module_variable_dict["A.var_trainable"],
    "B.var_fixed": example_tf_module_variable_dict["A.var_fixed"],
}

kernel_param_dict = {
    "SquaredExponential.lengthscales": {
        "value": Data.ls,
        "trainable": False,
        "shape": (),
    },
    "SquaredExponential.variance": {"value": Data.var, "trainable": True, "shape": ()},
}

compose_kernel_param_dict = {
    "kernels[0].kernels[0].variance": kernel_param_dict["SquaredExponential.variance"],
    "kernels[0].kernels[0].lengthscales": kernel_param_dict["SquaredExponential.lengthscales"],
    "kernels[0].kernels[1].variance": kernel_param_dict["SquaredExponential.variance"],
    "kernels[0].kernels[1].lengthscales": kernel_param_dict["SquaredExponential.lengthscales"],
    "kernels[1].kernels[0].variance": kernel_param_dict["SquaredExponential.variance"],
    "kernels[1].kernels[0].lengthscales": kernel_param_dict["SquaredExponential.lengthscales"],
    "kernels[1].kernels[1].variance": kernel_param_dict["SquaredExponential.variance"],
    "kernels[1].kernels[1].lengthscales": kernel_param_dict["SquaredExponential.lengthscales"],
}

model_gp_param_dict = {
    "kernel.lengthscales": kernel_param_dict["SquaredExponential.lengthscales"],
    "kernel.variance": kernel_param_dict["SquaredExponential.variance"],
    "likelihood.variance": {"value": 1.0, "trainable": True, "shape": ()},
    "inducing_variable.Z": {
        "value": Data.Z,
        "trainable": True,
        "shape": (Data.M, Data.D),
    },
    "SVGP.q_mu": {
        "value": np.zeros((Data.M, 1)),
        "trainable": False,
        "shape": (Data.M, 1),
    },
    "SVGP.q_sqrt": {
        "value": np.ones((Data.M, 1)),
        "trainable": True,
        "shape": (Data.M, 1),
    },
}

example_dag_module_param_dict = {
    "SVGP.kernel.variance\nSVGP.kernel.lengthscales": kernel_param_dict[
        "SquaredExponential.lengthscales"
    ],
    "SVGP.likelihood.variance": {"value": 1.0, "trainable": True, "shape": ()},
    "SVGP.inducing_variable.Z": {
        "value": Data.Z,
        "trainable": True,
        "shape": (Data.M, Data.D),
    },
    "SVGP.q_mu": {
        "value": np.zeros((Data.M, 1)),
        "trainable": False,
        "shape": (Data.M, 1),
    },
    "SVGP.q_sqrt": {
        "value": np.ones((Data.M, 1)),
        "trainable": True,
        "shape": (Data.M, 1),
    },
}

compose_kernel_param_print_string = """\
name                                        class      transform    prior    trainable    shape    dtype      value\n\
------------------------------------------  ---------  -----------  -------  -----------  -------  -------  -------\n\
Product.kernels[0].kernels[0].variance      Parameter  Softplus              True         ()       float64        1\n\
Product.kernels[0].kernels[0].lengthscales  Parameter  Softplus              False        ()       float64        2\n\
Product.kernels[0].kernels[1].variance      Parameter  Softplus              True         ()       float64        1\n\
Product.kernels[0].kernels[1].lengthscales  Parameter  Softplus              False        ()       float64        2\n\
Product.kernels[1].kernels[0].variance      Parameter  Softplus              True         ()       float64        1\n\
Product.kernels[1].kernels[0].lengthscales  Parameter  Softplus              False        ()       float64        2\n\
Product.kernels[1].kernels[1].variance      Parameter  Softplus              True         ()       float64        1\n\
Product.kernels[1].kernels[1].lengthscales  Parameter  Softplus              False        ()       float64        2"""

kernel_param_print_string = """\
name                             class      transform    prior    trainable    shape    dtype      value\n\
-------------------------------  ---------  -----------  -------  -----------  -------  -------  -------\n\
SquaredExponential.variance      Parameter  Softplus              True         ()       float64        1\n\
SquaredExponential.lengthscales  Parameter  Softplus              False        ()       float64        2"""

kernel_param_print_string_with_shift = """\
name                             class      transform         prior    trainable    shape    dtype      value\n\
-------------------------------  ---------  ----------------  -------  -----------  -------  -------  -------\n\
SquaredExponential.variance      Parameter  Softplus + Shift           True         ()       float64        1\n\
SquaredExponential.lengthscales  Parameter  Softplus + Shift           False        ()       float64        2"""

model_gp_param_print_string = """\
name                      class      transform    prior    trainable    shape    dtype    value\n\
------------------------  ---------  -----------  -------  -----------  -------  -------  --------\n\
SVGP.kernel.variance      Parameter  Softplus              True         ()       float64  1.0\n\
SVGP.kernel.lengthscales  Parameter  Softplus              False        ()       float64  2.0\n\
SVGP.likelihood.variance  Parameter  Softplus              True         ()       float64  1.0\n\
SVGP.inducing_variable.Z  Parameter  Identity              True         (10, 1)  float64  [[0.5...\n\
SVGP.q_mu                 Parameter  Identity              False        (10, 1)  float64  [[0....\n\
SVGP.q_sqrt               Parameter  Softplus              True         (10, 1)  float64  [[1...."""

example_tf_module_variable_print_string = """\
name             class             transform    prior    trainable    shape      dtype    value\n\
---------------  ----------------  -----------  -------  -----------  ---------  -------  --------\n\
A.var_trainable  ResourceVariable                        True         (2, 2, 1)  float32  [[[0....\n\
A.var_fixed      ResourceVariable                        False        (2, 2, 1)  float32  [[[1...."""

example_module_list_variable_print_string = """\
name                                 class             transform    prior    trainable    shape      dtype    value\n\
-----------------------------------  ----------------  -----------  -------  -----------  ---------  -------  --------\n\
B.submodule_list[0].var_trainable    ResourceVariable                        True         (2, 2, 1)  float32  [[[0....\n\
B.submodule_list[0].var_fixed        ResourceVariable                        False        (2, 2, 1)  float32  [[[1....\n\
B.submodule_list[1].var_trainable    ResourceVariable                        True         (2, 2, 1)  float32  [[[0....\n\
B.submodule_list[1].var_fixed        ResourceVariable                        False        (2, 2, 1)  float32  [[[1....\n\
B.submodule_dict['a'].var_trainable  ResourceVariable                        True         (2, 2, 1)  float32  [[[0....\n\
B.submodule_dict['a'].var_fixed      ResourceVariable                        False        (2, 2, 1)  float32  [[[1....\n\
B.submodule_dict['b'].var_trainable  ResourceVariable                        True         (2, 2, 1)  float32  [[[0....\n\
B.submodule_dict['b'].var_fixed      ResourceVariable                        False        (2, 2, 1)  float32  [[[1....\n\
B.var_trainable                      ResourceVariable                        True         (2, 2, 1)  float32  [[[0....\n\
B.var_fixed                          ResourceVariable                        False        (2, 2, 1)  float32  [[[1...."""

# Note: we use grid format here because we have a double reference to the same variable
# which does not render nicely in the table formatting.
# The internal structure of the Keras model changed in TensorFlow versions 2.5.0 and 2.6.0.
example_tf_keras_model_tf_2_6_0 = """\
+-------------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+\n\
| name                          | class            | transform   | prior   | trainable   | shape     | dtype   | value    |\n\
+===============================+==================+=============+=========+=============+===========+=========+==========+\n\
| C._trainable_weights[0]       | ResourceVariable |             |         | True        | (2, 2, 1) | float32 | [[[0.... |\n\
| C.variable                    |                  |             |         |             |           |         |          |\n\
+-------------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+\n\
| C._trainable_weights[1]       | ResourceVariable |             |         | True        | ()        | float64 | 0.0      |\n\
+-------------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+\n\
| C._self_tracked_trackables[0] | Parameter        | Identity    |         | True        | ()        | float64 | 0.0      |\n\
| C.param                       |                  |             |         |             |           |         |          |\n\
+-------------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+"""


example_tf_keras_model_tf_2_5_0 = """\
+-------------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+\n\
| name                          | class            | transform   | prior   | trainable   | shape     | dtype   | value    |\n\
+===============================+==================+=============+=========+=============+===========+=========+==========+\n\
| C._trainable_weights[0]       | ResourceVariable |             |         | True        | (2, 2, 1) | float32 | [[[0.... |\n\
| C.variable                    |                  |             |         |             |           |         |          |\n\
+-------------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+\n\
| C._self_tracked_trackables[0] | Parameter        | Identity    |         | True        | ()        | float64 | 0.0      |\n\
| C.param                       |                  |             |         |             |           |         |          |\n\
+-------------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+"""

example_tf_keras_model = """\
+-------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+\n\
| name                    | class            | transform   | prior   | trainable   | shape     | dtype   | value    |\n\
+=========================+==================+=============+=========+=============+===========+=========+==========+\n\
| C._trainable_weights[0] | ResourceVariable |             |         | True        | (2, 2, 1) | float32 | [[[0.... |\n\
| C.variable              |                  |             |         |             |           |         |          |\n\
+-------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+\n\
| C.param                 | Parameter        | Identity    |         | True        | ()        | float64 | 0.0      |\n\
+-------------------------+------------------+-------------+---------+-------------+-----------+---------+----------+"""

if Version(tf.__version__) >= Version("2.6"):
    example_tf_keras_model = example_tf_keras_model_tf_2_6_0
elif Version(tf.__version__) >= Version("2.5"):
    example_tf_keras_model = example_tf_keras_model_tf_2_5_0

# ------------------------------------------
# Fixtures
# ------------------------------------------


@pytest.fixture(params=[A, B, create_kernel, create_model])
def module(request: SubRequest) -> Any:
    return request.param()


@pytest.fixture
def dag_module() -> gpflow.models.GPModel:
    dag = create_model()
    dag.kernel.variance = dag.kernel.lengthscales
    return dag


# ------------------------------------------
# Tests
# ------------------------------------------


def test_leaf_components_only_returns_parameters_and_variables(module: Any) -> None:
    for path, variable in leaf_components(module).items():
        assert isinstance(variable, tf.Variable) or isinstance(variable, gpflow.Parameter)


@pytest.mark.parametrize(
    "module_callable, expected_param_dicts",
    [(create_kernel, kernel_param_dict), (create_model, model_gp_param_dict)],
)
def test_leaf_components_registers_variable_properties(
    module_callable: Callable[[], Any], expected_param_dicts: Mapping[str, Any]
) -> None:
    module = module_callable()
    for path, variable in leaf_components(module).items():
        param_name = path.split(".")[-2] + "." + path.split(".")[-1]
        assert isinstance(variable, gpflow.Parameter)
        np.testing.assert_almost_equal(variable.numpy(), expected_param_dicts[param_name]["value"])
        assert variable.trainable == expected_param_dicts[param_name]["trainable"]
        assert variable.shape == expected_param_dicts[param_name]["shape"]


@pytest.mark.parametrize(
    "module_callable, expected_param_dicts",
    [
        (create_compose_kernel, compose_kernel_param_dict),
    ],
)
def test_leaf_components_registers_compose_kernel_variable_properties(
    module_callable: Callable[[], Any], expected_param_dicts: Mapping[str, Any]
) -> None:
    module = module_callable()
    leaf_components_dict = leaf_components(module)
    assert len(leaf_components_dict) > 0
    for path, variable in leaf_components_dict.items():
        path_as_list = path.split(".")
        param_name = path_as_list[-3] + "." + path_as_list[-2] + "." + path_as_list[-1]
        assert isinstance(variable, gpflow.Parameter)
        np.testing.assert_almost_equal(variable.numpy(), expected_param_dicts[param_name]["value"])
        assert variable.trainable == expected_param_dicts[param_name]["trainable"]
        assert variable.shape == expected_param_dicts[param_name]["shape"]


@pytest.mark.parametrize(
    "module_class, expected_var_dicts",
    [
        (A, example_tf_module_variable_dict),
        (B, example_module_list_variable_dict),
    ],
)
def test_leaf_components_registers_param_properties(
    module_class: Type[Any], expected_var_dicts: Mapping[str, Any]
) -> None:
    module = module_class()
    for path, variable in leaf_components(module).items():
        var_name = path.split(".")[-2] + "." + path.split(".")[-1]
        assert isinstance(variable, tf.Variable)
        np.testing.assert_equal(variable.numpy(), expected_var_dicts[var_name]["value"])
        assert variable.trainable == expected_var_dicts[var_name]["trainable"]
        assert variable.shape == expected_var_dicts[var_name]["shape"]


@pytest.mark.parametrize("expected_var_dicts", [example_dag_module_param_dict])
def test_merge_leaf_components_merges_keys_with_same_values(
    dag_module: Any, expected_var_dicts: Mapping[str, Any]
) -> None:
    leaf_components_dict = leaf_components(dag_module)
    for path, variable in _merge_leaf_components(leaf_components_dict).items():
        assert path in expected_var_dicts
        for sub_path in path.split("\n"):
            assert sub_path in leaf_components_dict
            assert leaf_components_dict[sub_path] is variable


@pytest.mark.parametrize(
    "module_callable, expected_param_print_string",
    [
        (create_compose_kernel, compose_kernel_param_print_string),
        (create_kernel, kernel_param_print_string),
        (create_model, model_gp_param_print_string),
        (A, example_tf_module_variable_print_string),
        (B, example_module_list_variable_print_string),
    ],
)
def test_print_summary_output_string(
    module_callable: Callable[[], Any], expected_param_print_string: str
) -> None:
    with as_context(Config(positive_minimum=0.0)):
        assert tabulate_module_summary(module_callable()) == expected_param_print_string


def test_print_summary_output_string_with_positive_minimum() -> None:
    with as_context(Config(positive_minimum=1e-6)):
        assert tabulate_module_summary(create_kernel()) == kernel_param_print_string_with_shift


def test_print_summary_for_keras_model() -> None:
    # Note: best to use `grid` formatting for `tf.keras.Model` printing
    # because of the duplicates in the references to the variables.
    assert tabulate_module_summary(C(), tablefmt="grid") == example_tf_keras_model


def test_leaf_components_combination_kernel() -> None:
    """
    Regression test for kernel compositions - output for printing should not be empty (issue #1066).
    """
    k = gpflow.kernels.SquaredExponential() + gpflow.kernels.SquaredExponential()
    assert leaf_components(k), "Combination kernel should have non-empty leaf components"


def test_module_parameters_return_iterators_not_generators() -> None:
    """
    Regression test: Ensure that gpflow.Module parameters return iterators like in TF2, not
    generators.

    Reason:
    param = m.params  # <generator object>
    x = [p for p in param] # List[Parameters]
    y = [p for p in param] # [] empty!
    """
    m = create_model()
    assert isinstance(m, gpflow.base.Module)
    assert isinstance(m.parameters, tuple)
    assert isinstance(m.trainable_parameters, tuple)
