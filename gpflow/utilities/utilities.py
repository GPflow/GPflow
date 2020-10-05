# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from packaging.version import Version
from tabulate import tabulate

from ..base import Parameter
from ..config import default_float, default_int, default_summary_fmt
from .ops import cast

__all__ = [
    "set_trainable",
    "multiple_assign",
    "training_loop",
    "print_summary",
    "tabulate_module_summary",
    "deepcopy",
    "freeze",
    "leaf_components",
    "parameter_dict",
    "read_values",
    "to_default_float",
    "to_default_int",
    "reset_cache_bijectors",
    "select_dict_parameters_with_prior",
]

TraverseInput = TypeVar("TraverseInput", tf.Variable, tf.Module, Parameter)
State = Any
Path = str
Accumulator = Tuple[Path, State]
TraverseUpdateCallable = Callable[[TraverseInput, Path, State], State]


def to_default_int(x):
    return cast(x, dtype=default_int())


def to_default_float(x):
    return cast(x, dtype=default_float())


def set_trainable(model: Union[tf.Module, Iterable[tf.Module]], flag: bool) -> None:
    """
    Set trainable flag for all `tf.Variable`s and `gpflow.Parameter`s in a `tf.Module` or collection
    of `tf.Module`s.
    """
    modules = [model] if isinstance(model, tf.Module) else model

    for mod in modules:
        for variable in mod.variables:
            variable._trainable = flag


def multiple_assign(module: tf.Module, parameters: Dict[str, tf.Tensor]):
    """
    Multiple assign takes a dictionary with new values. Dictionary keys are paths to the
    `tf.Variable`s or `gpflow.Parameter` of the input module.

    :param module: `tf.Module`.
    :param parameters: a dictionary with keys of the form ".module.path.to.variable" and new value tensors.
    """
    reference_var_dict = parameter_dict(module)
    for path, value in parameters.items():
        reference_var_dict[path].assign(value)


def read_values(module: tf.Module) -> Dict[str, np.ndarray]:
    """Returns a dictionary of numpy values of the module parameters (variables)."""
    return {k: v.numpy() for k, v in parameter_dict(module).items()}


def parameter_dict(module: tf.Module) -> Dict[str, Union[Parameter, tf.Variable]]:
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


def training_loop(
    closure: Callable[[], tf.Tensor],
    optimizer: Optional[tf.optimizers.Optimizer] = None,
    var_list: List[tf.Variable] = None,
    maxiter=1e3,
    compile=False,
):
    """
    Simple generic training loop. At each iteration uses a GradientTape to compute
    the gradients of a loss function with respect to a set of variables.

    :param closure: Callable that constructs a loss function based on data and model being trained
    :param optimizer: tf.optimizers or tf.keras.optimizers that updates variables by applying the
        corresponding loss gradients. Adam is a default optimizer with default settings.
    :param var_list: List of model variables to be learnt during training
    :param maxiter: Maximum number of
    :return:
    """

    optimizer = tf.optimizers.Adam() if optimizer is None else optimizer

    def optimization_step():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(var_list)
            loss = closure()
        grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

    if compile:
        optimization_step = tf.function(optimization_step)

    for _ in range(int(maxiter)):
        optimization_step()


def print_summary(module: tf.Module, fmt: str = None):
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
    def get_transform(path, var):
        if hasattr(var, "transform") and var.transform is not None:
            if isinstance(var.transform, tfp.bijectors.Chain):
                return " + ".join(b.__class__.__name__ for b in var.transform.bijectors[::-1])
            return var.transform.__class__.__name__
        return None

    def get_prior(path, var):
        if hasattr(var, "prior") and var.prior is not None:
            return var.prior.name
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
    return tabulate(column_values, headers=column_names, tablefmt=tablefmt)


def leaf_components(input: tf.Module):
    return _get_leaf_components(input)


def _merge_leaf_components(
    input: Dict[str, Union[tf.Variable, tf.Tensor, Parameter]]
) -> Dict[str, Union[tf.Variable, tf.Tensor, Parameter]]:

    ref_fn = lambda x: (x if isinstance(x, Parameter) else x.ref())
    deref_fn = lambda x: (x if isinstance(x, Parameter) else x.deref())

    input_values = set([ref_fn(value) for value in input.values()])
    if len(input_values) == len(input):
        return input

    tmp_dict = dict()  # Type: Dict[ref, str]
    for key, value in input.items():
        ref = ref_fn(value)
        if ref in tmp_dict:
            tmp_dict[ref] = f"{tmp_dict[ref]}\n{key}"
        else:
            tmp_dict[ref] = key
    return {key: deref_fn(ref) for ref, key in tmp_dict.items()}


def _get_leaf_components(input_module: tf.Module):
    """
    Returns a list of tuples each corresponding to a gpflow.Parameter or tf.Variable in the each
    submodules of a given tf.Module. Each tuple consists of an specific Parameter (or Variable) and
    its relative path inside the module, which is constructed recursively by adding a prefix with
    the path to the current module. Designed to be used as a helper for the method 'print_summary'.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :return:
    """
    target_types = (Parameter, tf.Variable)
    input_name, state = input_module.__class__.__name__, dict()
    accumulator = (input_name, state)

    def update_state(parameter_or_variable, path, state):
        state[path] = parameter_or_variable
        return state

    state = traverse_module(input_module, accumulator, update_state, target_types)
    return state


def reset_cache_bijectors(input_module: tf.Module) -> tf.Module:
    """
    Recursively finds tfp.bijectors.Bijector-s inside the components of the tf.Module using `traverse_component`.
    Resets the caches stored inside each tfp.bijectors.Bijector.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :return:
    """
    if Version(tfp.__version__) >= Version("0.11.0"):
        if hasattr(tfp.bijectors.Identity()._cache, "clear"):
            # implementation in `master` branch (checked 29 Sep 2020) provides clear():

            def _clear_bijector_cache(bijector: tfp.bijectors.Bijector):
                bijector._cache.clear()

        else:
            # previous versions (including the versions 0.11.0 and 0.11.1 released as of 29 Sep 2020) provide reset(), but its implementation is broken

            def _clear_bijector_cache(bijector: tfp.bijectors.Bijector):
                # workaround for broken implementation of bijector._cache.reset():
                cache = bijector._cache
                cache_type = type(cache.forward)
                assert type(cache.inverse) == cache_type
                cache.__init__(cache.forward._func, cache.inverse._func, cache_type)

    else:
        # fallback for backwards-compatibility with tensorflow_probability < 0.11.0

        def _clear_bijector_cache(bijector: tfp.bijectors.Bijector):
            # `_from_x` and `_from_y` are cache dictionaries for forward and inverse transformations
            bijector._from_x.clear()
            bijector._from_y.clear()

    target_types = (tfp.bijectors.Bijector,)
    accumulator = ("", None)

    def clear_bijector(bijector, _, state):
        if not isinstance(bijector, tfp.bijectors.Bijector):
            return  # skip submodules that are not bijectors

        _clear_bijector_cache(bijector)

        if isinstance(bijector, tfp.bijectors.Chain):
            # recursively clear caches of sub-bijectors
            for m in bijector.submodules:
                if isinstance(m, tfp.bijectors.Bijector):
                    _clear_bijector_cache(m)

        return state

    _ = traverse_module(input_module, accumulator, clear_bijector, target_types)
    return input_module


M = TypeVar("M", bound=tf.Module)


def deepcopy(input_module: M, memo: Optional[Dict[int, Any]] = None) -> M:
    """
    Returns a deepcopy of the input tf.Module. To do that first resets the caches stored inside each
    tfp.bijectors.Bijector to allow the deepcopy of the tf.Module.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :param memo: passed through to func:`copy.deepcopy` (see https://docs.python.org/3/library/copy.html).
    :return: Returns a deepcopy of an input object.
    """
    return copy.deepcopy(reset_cache_bijectors(input_module), memo)


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
    m: TraverseInput, acc: Accumulator, update_cb: TraverseUpdateCallable, target_types: tuple
) -> Accumulator:
    """
    Recursively traverses `m`, accumulating in `acc` a path and a state until it finds an object of type
    in `target_types` to apply `update_cb` to update the accumulator `acc` and/or the object.

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


def select_dict_parameters_with_prior(model: tf.Module) -> Dict[str, Parameter]:
    """Collects parameters with prior into a dictionary."""
    return {
        k: p
        for k, p in parameter_dict(model).items()
        if hasattr(p, "prior") and p.prior is not None
    }
