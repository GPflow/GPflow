import re
from copy import deepcopy
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union, TypeVar, Any, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tabulate import tabulate

from tensorflow.python.training.tracking.data_structures import ListWrapper, _DictWrapper

from ..base import Parameter
from ..config import summary_fmt

__all__ = [
    "set_trainable",
    "multiple_assign",
    "training_loop",
    "print_summary",
    "deepcopy_components"
]

TraverseInput = TypeVar("TraverseInput", tf.Variable, tf.Module, Parameter)
State = Any
Path = str
Accumulator = Tuple[Path, State]
TraverseUpdateCallable = Callable[[TraverseInput, Path, State], State]


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


def training_loop(closure: Callable[..., tf.Tensor],
                  optimizer: Optional[tf.optimizers.Optimizer] = None,
                  var_list: List[tf.Variable] = None,
                  maxiter=1e3,
                  jit=False):
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
        with tf.GradientTape() as tape:
            tape.watch(var_list)
            loss = closure()
        grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

    if jit:
        optimization_step = tf.function(optimization_step)

    for _ in range(int(maxiter)):
        optimization_step()


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
    input_name, state = input.__class__.__name__, dict()
    accumulator = (input_name, state)

    def update_state(parameter_or_variable, path, state):
        state[path] = parameter_or_variable
        return state

    state = traverse_module(input_module, accumulator, update_state, target_types)
    return state


def deepcopy_components(input: tf.Module):
    """
    Returns a deepcopy of the input tf.Module. Recursively finds tfp.bijectors.Bijector-s
    inside the components of the tf.Module using `traverse_component`. Resets the caches stored inside each
    tfp.bijectors.Bijector to allow the deepcopy of the tf.Module.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :return:
    """
    target_types = (tfp.bijectors.Bijector,)
    accumulator = ('', None)

    def clear_bijector(bijector, _, state):
        bijector._from_x.clear()
        bijector._from_y.clear()
        return state

    _ = traverse_module(input, accumulator, clear_bijector, target_types)
    return deepcopy(input_module)


def traverse_module(m: TraverseInput,
                    acc: Accumulator,
                    update_cb: TraverseUpdateCallable,
                    target_types: tuple) -> Accumulator:
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

    if isinstance(m, ListWrapper):
        for term_idx, subterm in enumerate(m):
            new_acc = (f"{path}[{term_idx}]", new_state)
            new_state = traverse_module(subterm, new_acc, update_cb, target_types)
    elif isinstance(m, _DictWrapper):
        for term_idx, subterm in m.items():
            new_acc = (f"{path}['{term_idx}']", new_state)
            new_state = traverse_module(subterm, new_acc, update_cb, target_types)
    elif isinstance(m, tf.Module):
        for name, submodule in vars(m).items():
            if name in tf.Module._TF_MODULE_IGNORED_PROPERTIES:
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
