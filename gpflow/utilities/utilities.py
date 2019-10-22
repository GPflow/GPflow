import re
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

from ..base import Parameter
from ..config import summary_fmt

__all__ = [
    "set_trainable",
    "multiple_assign",
    "print_summary"
]


def set_trainable(model: tf.Module, flag: bool):
    """
    Set trainable flag for all `tf.Variable`s and `gpflow.Parameter`s in a module.
    """
    for variable in model.trainable_variables:
        variable._trainable = flag


def multiple_assign(input: tf.Module, vars_dict: Dict[str, tf.Tensor]):
    """
    Multiple assign takes a dictionary with new values. Dictionary keys are paths to the
    `tf.Variable`s or `gpflow.Parameters` of the input module.

    :param input: `tf.Module`.
    :param vars_dict: a dictionary with keys of the form "module.path.to.variable" and new value tensors.
    """
    reference_var_dict = leaf_components(input)
    for path, value in vars_dict.items():
        reference_var_dict[path].assign(value)


def print_summary(module: tf.Module, fmt: str = None):
    """
    Prints a summary of the parameters and variables contained in a tf.Module.
    """

    if fmt == "notebook":
        from IPython.core.display import display, HTML
        tab = tabulate_module_summary(module, "html")
        display(HTML(tab))
    else:
        print(tabulate_module_summary(module, fmt))


def tabulate_module_summary(module: tf.Module, tablefmt: str = None) -> str:
    tablefmt = tablefmt if tablefmt is not None else summary_fmt()
    column_names = ['name', 'class', 'transform', 'trainable', 'shape', 'dtype', 'value']

    def get_name(v):
        return v.__class__.__name__

    def get_transform(v):
        if hasattr(v, "transform") and v.transform is not None:
            return v.transform.__class__.__name__
        return None

    merged_leaf_components = _merge_leaf_components(leaf_components(module))

    column_values = [[
        path,
        get_name(variable),
        get_transform(variable),
        variable.trainable,
        variable.shape,
        variable.dtype.name,
        _str_tensor_value(variable.numpy())
    ] for path, variable in merged_leaf_components.items()]
    return tabulate(column_values, headers=column_names, tablefmt=tablefmt)


def leaf_components(input: tf.Module):
    return _get_leaf_components(input)


def _merge_leaf_components(
        input: Dict[str, Union[tf.Tensor, Parameter]]) -> Dict[str, Union[tf.Tensor, Parameter]]:
    input_values = set(
        [value.experimental_ref() if isinstance(value, tf.Variable) else value for value in input.values()]
    )
    if len(input_values) == len(input):
        return input
    tmp_dict = dict()
    for key, item in input.items():
        if item in tmp_dict:
            tmp_dict[item] = f"{tmp_dict[item]}\n{key}"
        else:
            tmp_dict[item] = key
    return {key: item for item, key in tmp_dict.items()}


def _get_leaf_components(input: tf.Module, prefix: Optional[str] = None):
    """
    Returns a list of tuples each corresponding to a gpflow.Parameter or tf.Variable in the each
    submodules of a given tf.Module. Each tuple consists of an specific Parameter (or Variable) and
    its relative path inside the module, which is constructed recursively by adding a prefix with
    the path to the current module. Designed to be used as a helper for the method 'print_summary'.

    :param module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :param prefix: string containing the relative path to module, by default set to None.
    :return:
    """
    if not isinstance(input, tf.Module):
        raise TypeError("Input object expected to have `tf.Module` type")

    prefix = input.__class__.__name__ if prefix is None else prefix
    var_dict = dict()

    for key, submodule in vars(input).items():
        if key in tf.Module._TF_MODULE_IGNORED_PROPERTIES:
            continue
        elif isinstance(submodule, Parameter) or isinstance(submodule, tf.Variable):
            var_dict[f"{prefix}.{key}"] = submodule
        elif isinstance(submodule, tf.Module):
            submodule_var = _get_leaf_components(submodule, prefix=f"{prefix}.{key}")
            var_dict.update(submodule_var)
        elif isinstance(submodule, ListWrapper):
            submodule_name = input.__class__.__name__
            for term_idx, subterm in enumerate(submodule):
                subterm_key = f"{submodule_name}_{key}[{term_idx}]"
                if isinstance(subterm, tf.Module):
                    subterm_var = _get_leaf_components(subterm, prefix=f"{prefix}.{subterm_key}")
                    var_dict.update(subterm_var)
        elif isinstance(submodule, _DictWrapper):
            submodule_name = input.__class__.__name__
            for term_key, subterm in submodule.items():
                subterm_key = f"{submodule_name}_{key}[{term_key}]"
                subterm_var = _get_leaf_components(subterm, prefix=f"{prefix}.{subterm_key}")
                var_dict.update(subterm_var)
    return var_dict


@lru_cache()
def _first_three_elements_regexp():
    num_re = r"[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
    pat_re = rf"^(?:(\[+)\s*)?({num_re})(?:\s+({num_re})(?:\s+({num_re}))?)?.*?"
    return re.compile(pat_re)


def _str_tensor_value(value: np.ndarray):
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
