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

import re
from functools import lru_cache
from typing import Optional

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from .. import Parameter


def print_summary(module: tf.Module, fmt: str = None):
    """
    Prints a summary of the parameters and variables contained in a tf.Module and its components.
    """
    fmt = fmt if fmt is not None else "simple"
    column_names = ['name', 'class', 'transform', 'trainable', 'shape', 'dtype', 'value']

    def get_name(v):
        return v.__class__.__name__

    def get_transform(v):
        if hasattr(v, 'transform') and v.transform is not None:
            return v.transform.__class__.__name__
        return None

    column_values = [[
        path,
        get_name(variable),
        get_transform(variable),
        variable.trainable,
        variable.shape,
        variable.dtype.name,
        get_str_tensor_value(variable.numpy())
    ] for path, variable in get_component_variables(module).items()]

    print(tabulate(column_values, headers=column_names, tablefmt=fmt))


def get_component_variables(module: tf.Module, prefix: Optional[str] = None):
    """
    Returns a list of tuples each corresponding to a gpflow.Parameter or tf.Variable in the each
    submodules of a given tf.Module. Each tuple consists of an specific Parameter (or Variable) and
    its relative path inside the module, which is constructed recursively by adding a prefix with
    the path to the current module. Designed to be used as a helper for the method 'print_summary'.

    :param module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :param prefix: string containing the relative path to module, by default set to None.
    :return:
    """
    prefix = module.__class__.__name__ if prefix is None else prefix
    var_dict = {}
    if isinstance(module, tf.keras.Model) or isinstance(module, tf.keras.layers.Layer):
        for weight in module.weights:
            if weight in module.non_trainable_weights:
                weight._trainable = False
            weight_path = '{}.{}'.format(prefix, _format_keras_weight_names(weight.name))
            var_dict[weight_path] = weight
    else:
        module_dict = vars(module)
        for key, submodule in module_dict.items():
            if key in tf.Module._TF_MODULE_IGNORED_PROPERTIES:
                continue
            elif isinstance(submodule, Parameter) or isinstance(submodule, tf.Variable):
                var_dict['{}.{}'.format(prefix, key)] = submodule
            elif isinstance(submodule, tf.Module):
                submodule_var = get_component_variables(submodule,
                                                        prefix='{}.{}'.format(prefix, key))
                var_dict.update(submodule_var)

    return var_dict


def _format_keras_weight_names(name: str):
    return re.sub('/', '.', name).strip(':0')


@lru_cache()
def _first_three_elements_regexp():
    num_re = r"[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
    pat_re = rf"^(?:(\[+)\s*)?({num_re})(?:\s+({num_re})(?:\s+({num_re}))?)?.*?"
    return re.compile(pat_re)


def get_str_tensor_value(value: np.ndarray):
    value_str = str(value)
    if value.size <= 3:
        return value_str

    max_chars = 500
    value_str = value_str[:max_chars]
    regexp = _first_three_elements_regexp()
    match = regexp.match(value_str)
    assert match is not None
    brackets, elem1, elem2, elem3 = match.groups()

    out = f"{elem1}"
    if elem2 is not None:
        out = f"{out}{f', {elem2}'}"
        if elem3 is not None:
            out = f"{out}{f', {elem3}'}"
    if brackets is not None:
        out = f"{brackets}{out}..."

    return out
