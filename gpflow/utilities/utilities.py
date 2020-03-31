import copy
import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tabulate import tabulate

from .ops import cast
from ..base import Parameter, Module
from ..config import default_float, default_int, default_summary_fmt

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
    "getattr_by_path",
    "setattr_by_path",
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


def set_trainable(model: tf.Module, flag: bool):
    """
    Set trainable flag for all `tf.Variable`s and `gpflow.Parameter`s in a module.
    """
    for variable in model.variables:
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
        from IPython.core.display import display, HTML

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

    input_values = set([value.experimental_ref() for value in input.values()])
    if len(input_values) == len(input):
        return input
    tmp_dict = dict()  # Type: Dict[ref, str]
    for key, variable in input.items():
        ref = variable.experimental_ref()
        if ref in tmp_dict:
            tmp_dict[ref] = f"{tmp_dict[ref]}\n{key}"
        else:
            tmp_dict[ref] = key
    return {key: ref.deref() for ref, key in tmp_dict.items()}


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
    target_types = (tfp.bijectors.Bijector,)
    accumulator = ("", None)

    def clear_cache(b):
        if isinstance(b, tfp.bijectors.Bijector):
            # `_from_x` and `_from_y` are cache dictionaries for forward and inverse transformations
            # in bijector class.
            b._from_x.clear()
            b._from_y.clear()

    def clear_bijector(bijector, _, state):
        clear_cache(bijector)
        if isinstance(bijector, tfp.bijectors.Chain):
            for m in bijector.submodules:
                clear_cache(m)
        return state

    _ = traverse_module(input_module, accumulator, clear_bijector, target_types)
    return input_module


def _get_by_name_index(parent: object, attr_str: str, index_str: Union[str, None]) -> object:
    attr = getattr(parent, attr_str)
    if index_str is not None:
        index = int(index_str)
        return attr[index]
    return attr


def _set_by_name_index(parent: object, value: Any, attr_str: str, index_str: Union[str, None]):
    if index_str is not None:
        index = int(index_str)
        attr = getattr(parent, attr_str)
        attr[index] = value
        # NOTE: tensorflow's __setattr__ override does not contain a check for the case where
        # an attribute holding a trackable object (e.g. a Variable) is being reassigned
        # a non-trackable object (e.g. a constant). Therefore, tensorflow still stores
        # internal references to the trackable object after reassignment, which can lead
        # to various problems, for example https://github.com/GPflow/GPflow/pull/1338
        # By calling delattr first, we are forcing tensorflow to remove its internal reference.
        # This is a tensorflow bug, see https://github.com/tensorflow/tensorflow/issues/37806
        delattr(parent, attr_str)
        setattr(parent, attr_str, attr)
    else:
        delattr(parent, attr_str)  # as above
        setattr(parent, attr_str, value)


def _get_last_attr_spec(parent: object, attr_path: str) -> Tuple[object, str, str]:
    """
    Returns second to last attribute, the next attribute name, and
    an index if there is a list access.

    Example:
        module = ModuleWithNestedStructure(...)
        attr_path = "a.b.c[0].d[1000]"
        value = module.a.b.c[0].d[1000]
        c0, dname, dindex = _get_last_attr_spec(module, "a.b.c[0].d[1000]")
        assert c0 is module.a.b.c[0]
        assert dname == "d"
        assert dindex == "1000"

    :param parent: A python object with a nested structure.
    :param attr_path: An attribute path.

    :returns: The value stored in the nested object by the attribute path,
        and last attribute name with an optional index.
    """

    # Regexp extracts attribute name and index if available.
    # - '(\w+)' is a group for attribute name.
    # - '(\[\s*(-?\d+)\s*\])?' is the index group and has a subgroup '(-?\d+)',
    #   that matches a number. This group may not appear in the search string.
    regexp = re.compile(r"^(\w+)(\[\s*(-?\d+)\s*\])?$")

    def parse(token: str) -> Tuple[str, str]:
        m = regexp.match(token)
        if m is None:
            raise ValueError(f"Cannot parse attribute path '{attr_path}'")
        attr_token, _, index_token = m.groups()
        return attr_token, index_token

    curr = parent
    tokens = attr_path.split(".")
    for group_token in tokens[:-1]:
        attr_token, index_token = parse(group_token)
        curr = _get_by_name_index(curr, attr_token, index_token)

    attr_token, index_token = parse(tokens[-1])
    return curr, attr_token, index_token


def getattr_by_path(target: object, attr_path: str) -> Any:
    """Get a value of nested attribute by a string path.

    :param target: Root object that contains nested attribute
    :param attr_path: String type value that represents path to an attribute.
    :return: Object stored at `attr_path`.

    Example:
        k = gpflow.kernels.Matern52()
        m = gpflow.models.GPR(..., kernel=kernel)
        lengthscales = getattr_by_path(m, "kernel.lengthscales")
    """
    try:
        descendant, attr, index = _get_last_attr_spec(target, attr_path)
        return _get_by_name_index(descendant, attr, index)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Cannot get value at path '{attr_path}'") from error


def setattr_by_path(target: object, attr_path: str, value: Any):
    """Set `value` by a given path to a nested attribute.

    :param target: Root object that contains a nested attribute.
    :param attr_path: String type value that represents path to the attribute.
    :param value: Value to assign to the attribute.

    Example:
        k = gpflow.kernels.Matern52()
        m = gpflow.models.GPR(..., kernel=kernel)
        setattr_by_path(m, "kernel.lengthscales", tf.constant(1.0, dtype=...))
    """
    try:
        descendant, attr, index = _get_last_attr_spec(target, attr_path)
        _set_by_name_index(descendant, value, attr, index)
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Cannot assign value at path '{attr_path}'") from error


M = TypeVar("M", bound=tf.Module)


def deepcopy(input_module: M) -> M:
    """
    Returns a deepcopy of the input tf.Module. To do that first resets the caches stored inside each
    tfp.bijectors.Bijector to allow the deepcopy of the tf.Module.

    :param input_module: tf.Module including keras.Model, keras.layers.Layer and gpflow.Module.
    :return: Returns a deepcopy of an input object.
    """
    return copy.deepcopy(reset_cache_bijectors(input_module))


def freeze(input_module: M) -> M:
    """
    Returns a deepcopy of the input tf.Module with constants instead of variables and parameters.

    :param input_module: tf.Module or gpflow.Module.
    :return: Returns a frozen deepcopy of an input object.
    """
    module_copy = deepcopy(input_module)
    for name, value in parameter_dict(module_copy).items():
        tensor = tf.convert_to_tensor(value, dtype=value.dtype)
        const_value = tf.constant(tensor)
        attr_path = name.lstrip(".")
        setattr_by_path(module_copy, attr_path, const_value)
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


def select_dict_parameters_with_prior(model: tf.Module) -> Dict[str, Parameter]:
    """Collects parameters with prior into a dictionary."""
    return {
        k: p
        for k, p in parameter_dict(model).items()
        if hasattr(p, "prior") and p.prior is not None
    }
