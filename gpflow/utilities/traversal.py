# Copyright 2017-2021 The GPflow Contributors. All Rights Reserved.
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

import copy
import re
from functools import lru_cache
from typing import Any, Callable, Dict, Mapping, Optional, Pattern, Tuple, Type, TypeVar, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from packaging.version import Version
from tabulate import tabulate
from tensorflow.python.util.object_identity import Reference

from ..base import AnyNDArray, Parameter
from ..config import default_summary_fmt

__all__ = [
    "deepcopy",
    "freeze",
    "leaf_components",
    "multiple_assign",
    "parameter_dict",
    "print_summary",
    "read_values",
    "reset_cache_bijectors",
    "select_dict_parameters_with_prior",
    "tabulate_module_summary",
]
LeafComponent = Union[tf.Variable, tf.Tensor, Parameter]
LeafVariable = Union[tf.Variable, Parameter]
HashableTensor = Union[Reference, Parameter]
TraverseInput = TypeVar("TraverseInput", tf.Variable, tf.Module, Parameter)
State = TypeVar("State")
Path = str
Accumulator = Tuple[Path, State]
TraverseUpdateCallable = Callable[[TraverseInput, Path, State], State]


def multiple_assign(module: tf.Module, parameters: Mapping[Path, tf.Tensor]) -> None:
    """
    Multiple assign takes a dictionary with new values. Dictionary keys are paths to the
    `tf.Variable`s or `gpflow.Parameter` of the input module.

    :param module: `tf.Module`.
    :param parameters: a dictionary with keys of the form ".module.path.to.variable" and new value tensors.
    """
    reference_var_dict = parameter_dict(module)
    for path, value in parameters.items():
        reference_var_dict[path].assign(value)


def read_values(module: tf.Module) -> Dict[Path, AnyNDArray]:
    """Returns a dictionary of numpy values of the module parameters (variables)."""
    return {k: v.numpy() for k, v in parameter_dict(module).items()}


def parameter_dict(module: tf.Module) -> Dict[Path, LeafVariable]:
    """
    Returns a dictionary of parameters (variables) for the `tf.Module` component.
    Dictionary keys are relative paths to the attributes to which parameters (variables) assigned to.

        class SubModule(tf.Module):
            def __init__(self):
                self.parameter = gpflow.Parameter(1.0)
                self.variable = tf.Variable(1.0)

        class Module(tf.Module):
            def __init__(self):
                self.submodule = SubModule()

        m = Module()
        params = parameter_dict(m)
        # {
        #   ".submodule.parameter": <parameter object>,
        #   ".submodule.variable": <variable object>
        # }
    """
    param_dict = leaf_components(module)
    return {f".{key.split('.', 1)[-1]}": value for key, value in param_dict.items()}


def print_summary(module: tf.Module, fmt: Optional[str] = None) -> None:
    """
    Prints a summary of the parameters and variables contained in a tf.Module.
    """
    fmt = fmt if fmt is not None else default_summary_fmt()
    if fmt == "notebook":
        from IPython.core.display import HTML, display

        tab = tabulate_module_summary(module, "html")
        display(HTML(tab))
    else:
        print(tabulate_module_summary(module, fmt))


def tabulate_module_summary(module: tf.Module, tablefmt: Optional[str] = None) -> str:
    def get_transform(path: Path, var: LeafComponent) -> Optional[str]:
        if hasattr(var, "transform") and var.transform is not None:
            if isinstance(var.transform, tfp.bijectors.Chain):
                return " + ".join(b.__class__.__name__ for b in var.transform.bijectors[::-1])
            return var.transform.__class__.__name__  # type: ignore[no-any-return]
        return None

    def get_prior(path: Path, var: LeafComponent) -> Optional[str]:
        if hasattr(var, "prior") and var.prior is not None:
            return var.prior.name  # type: ignore[no-any-return]
        return None

    # list of (column_name: str, column_getter: Callable[[tf.Variable], str]) tuples:
    column_definition = [
        ("name", lambda path, var: path),
        ("class", lambda path, var: var.__class__.__name__),
        ("transform", get_transform),
        ("prior", get_prior),
        ("trainable", lambda path, var: var.trainable),
        ("shape", lambda path, var: var.shape),
        ("dtype", lambda path, var: var.dtype.name),
        ("value", lambda path, var: _str_tensor_value(var.numpy())),
    ]
    column_names, column_getters = zip(*column_definition)

    merged_leaf_components = _merge_leaf_components(leaf_components(module))

    column_values = [
        [getter(path, variable) for getter in column_getters]
        for path, variable in merged_leaf_components.items()
    ]
    # mypy claims it's wrong to pass a `None` tablefmt below. I think `tabulate` has bad type hints.
    return tabulate(column_values, headers=column_names, tablefmt=tablefmt)  # type: ignore[arg-type]


def leaf_components(input: tf.Module) -> Mapping[Path, LeafVariable]:
    return _get_leaf_components(input)


def _merge_leaf_components(input: Mapping[Path, LeafComponent]) -> Mapping[Path, LeafComponent]:

    ref_fn: Callable[[LeafComponent], HashableTensor] = lambda x: (
        x if isinstance(x, Parameter) else x.ref()
    )
    deref_fn: Callable[[HashableTensor], LeafComponent] = lambda x: (
        x if isinstance(x, Parameter) else x.deref()
    )

    input_values = {ref_fn(value) for value in input.values()}
    if len(input_values) == len(input):
        return input

    tmp_dict: Dict[HashableTensor, Path] = {}
    for key, value in input.items():
        ref = ref_fn(value)
        if ref in tmp_dict:
            tmp_dict[ref] = f"{tmp_dict[ref]}\n{key}"
        else:
            tmp_dict[ref] = key
    return {key: deref_fn(ref) for ref, key in tmp_dict.items()}


def _get_leaf_components(input_module: tf.Module) -> Mapping[Path, LeafVariable]:
    """
    Returns a list of tuples each corresponding to a gpflow.Parameter or tf.Variable in the each
    submodules of a given tf.Module. Each tuple consists of an specific Parameter (or Variable) and
    its relative path inside the module, which is constructed recursively by adding a prefix with
    the path to the current module. Designed to be used as a helper for the method 'print_summary'.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :return:
    """
    target_types = (Parameter, tf.Variable)
    input_name = input_module.__class__.__name__
    state: Dict[Path, LeafVariable] = {}
    accumulator = (input_name, state)

    def update_state(
        parameter_or_variable: LeafVariable, path: Path, state: Dict[Path, LeafVariable]
    ) -> Dict[Path, LeafVariable]:
        state[path] = parameter_or_variable
        return state

    state = traverse_module(input_module, accumulator, update_state, target_types)
    return state


def reset_cache_bijectors(input_module: tf.Module) -> tf.Module:
    """
    Recursively finds tfp.bijectors.Bijector-s inside the components of the tf.Module using `traverse_component`.
    Resets the caches stored inside each tfp.bijectors.Bijector.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :returns: same object but with all bijector caches reset
    """
    if Version(tfp.__version__) >= Version("0.11.0"):
        if hasattr(tfp.bijectors.Identity()._cache, "clear"):
            # implementation in `master` branch (checked 29 Sep 2020) provides clear():

            def _clear_bijector_cache(bijector: tfp.bijectors.Bijector) -> None:
                bijector._cache.clear()

        else:  # pragma: no cover
            # previous versions (including the versions 0.11.0 and 0.11.1 released as of 29 Sep 2020) provide reset(), but its implementation is broken

            def _clear_bijector_cache(bijector: tfp.bijectors.Bijector) -> None:
                # workaround for broken implementation of bijector._cache.reset():
                cache = bijector._cache
                cache_type = type(cache.forward)
                assert type(cache.inverse) == cache_type
                cache.__init__(cache.forward._func, cache.inverse._func, cache_type)

    else:  # pragma: no cover
        # fallback for backwards-compatibility with tensorflow_probability < 0.11.0

        def _clear_bijector_cache(bijector: tfp.bijectors.Bijector) -> None:
            # `_from_x` and `_from_y` are cache dictionaries for forward and inverse transformations
            bijector._from_x.clear()
            bijector._from_y.clear()

    target_types = (tfp.bijectors.Bijector,)
    accumulator = ("", None)

    def clear_bijector(bijector: tfp.bijectors.Bijector, _: Path, state: None) -> None:
        if not isinstance(bijector, tfp.bijectors.Bijector):
            return  # skip submodules that are not bijectors

        _clear_bijector_cache(bijector)

        if isinstance(bijector, tfp.bijectors.Chain):
            # recursively clear caches of sub-bijectors
            for m in bijector.submodules:
                if isinstance(m, tfp.bijectors.Bijector):
                    _clear_bijector_cache(m)

        return state

    traverse_module(input_module, accumulator, clear_bijector, target_types)
    return input_module


M = TypeVar("M", bound=tf.Module)


def deepcopy(input_module: M, memo: Optional[Dict[int, Any]] = None) -> M:
    """
    Returns a deepcopy of the input tf.Module. To do that first resets the caches stored inside each
    tfp.bijectors.Bijector to allow the deepcopy of the tf.Module.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :param memo: passed through to func:`copy.deepcopy`
        (see https://docs.python.org/3/library/copy.html).
    :return: Returns a deepcopy of an input object.
    """
    return copy.deepcopy(reset_cache_bijectors(input_module), memo)  # type: ignore[no-any-return]


def freeze(input_module: M) -> M:
    """
    Returns a deepcopy of the input tf.Module with constants instead of variables and parameters.

    :param input_module: tf.Module or gpflow.Module.
    :return: Returns a frozen deepcopy of an input object.
    """
    objects_to_freeze = _get_leaf_components(input_module)
    memo_tensors = {id(v): tf.convert_to_tensor(v) for v in objects_to_freeze.values()}
    module_copy = deepcopy(input_module, memo_tensors)
    return module_copy


def traverse_module(
    m: TraverseInput,
    acc: Accumulator[State],
    update_cb: TraverseUpdateCallable[TraverseInput, State],
    target_types: Tuple[Type[Any], ...],
) -> State:
    """
    Recursively traverses `m`, accumulating in `acc` a path and a state until it finds an object of
    type in `target_types` to apply `update_cb` to update the accumulator `acc` and/or the object.

    :param m: tf.Module, tf.Variable or gpflow.Parameter
    :param acc: Tuple of path and state
    :param update_cb: Callable
    :param target_types: target class types
    :return:
    """
    path, state = acc

    new_state = state

    if isinstance(m, target_types):
        return update_cb(m, path, state)

    if isinstance(m, (list, tuple)):
        for term_idx, subterm in enumerate(m):
            new_acc = (f"{path}[{term_idx}]", new_state)
            new_state = traverse_module(subterm, new_acc, update_cb, target_types)
    elif isinstance(m, dict):
        for term_idx, subterm in m.items():
            new_acc = (f"{path}['{term_idx}']", new_state)
            new_state = traverse_module(subterm, new_acc, update_cb, target_types)
    elif isinstance(m, tf.Module):
        for name, submodule in vars(m).items():
            ignored_attributes = m._TF_MODULE_IGNORED_PROPERTIES
            # NOTE(awav): since tfp version 0.10.0, tfp.bijectors.Bijector instances have
            # `_parameters` dictionary with "self" references that cause
            # infinite recursive loop.
            if isinstance(m, tfp.bijectors.Bijector):
                ignored_attributes = ignored_attributes.union({"_parameters"})
            if name in ignored_attributes:
                continue
            new_acc = (f"{path}.{name}", new_state)
            new_state = traverse_module(submodule, new_acc, update_cb, target_types)
    return new_state


@lru_cache()
def _first_three_elements_regexp() -> Pattern[str]:
    num_re = r"[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
    pat_re = rf"^(?:(\[+)\s*)?({num_re})(?:\s+({num_re})(?:\s+({num_re}))?)?.*?"
    return re.compile(pat_re)


def _str_tensor_value(value: AnyNDArray) -> str:
    value_str = str(np.around(value, 5))
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


def select_dict_parameters_with_prior(model: tf.Module) -> Dict[Path, Parameter]:
    """Collects parameters with prior into a dictionary."""
    return {
        k: p
        for k, p in parameter_dict(model).items()
        if hasattr(p, "prior") and p.prior is not None
    }
