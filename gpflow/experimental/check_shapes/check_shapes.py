# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
"""
Tool for checking the shapes of function using tf Tensors.
"""
import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Match,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import tensorflow as tf

from ..utils import experimental

DimensionSpec = Union[int, str]
ShapeSpec = Sequence[DimensionSpec]
ArgumentSpec = Tuple[str, ShapeSpec]

Shaped = Union[np.ndarray, tf.Tensor]


_C = TypeVar("_C", bound=Callable)

# The special name used to represent the returned value. `return` is a good choice because we know
# that no argument can be called `return` because `return` a reserved keyword.
_RESULT_TOKEN = "return"

_NAME_RE_STR = "([_a-zA-Z][_a-zA-Z0-9]*)"
_ELLIPSIS_RE_STR = r"(\.\.\.)"
_VARIABLE_RE = re.compile(f"{_NAME_RE_STR}{_ELLIPSIS_RE_STR}?")
_ROOT_ARGUMENT_RE = re.compile(_NAME_RE_STR)
_ATTRIBUTE_ARGUMENT_RE = re.compile(f"\\.{_NAME_RE_STR}")
_INDEX_ARGUMENT_RE = re.compile(r"\[(\d+)\]")


class _ArgumentRef(ABC):
    """ A reference to an argument. """

    @property
    @abstractmethod
    def is_result(self) -> bool:
        """ Whether this is a reference to the function result. """

    def get(self, func: _C, arg_map: Mapping[str, Any]) -> Any:
        """ Get the value of this argument from this given map. """
        try:
            return self._get(arg_map)
        except Exception as e:
            raise ArgumentReferenceError(func, arg_map, self) from e

    @abstractmethod
    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        """ Get the value of this argument from this given map. """


class _RootArgumentRef(_ArgumentRef):
    """ A reference to a single argument. """

    def __init__(self, argument_name: str) -> None:
        self._argument_name = argument_name

    @property
    def is_result(self) -> bool:
        return self._argument_name == _RESULT_TOKEN

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        return arg_map[self._argument_name]

    def __repr__(self) -> str:
        return self._argument_name


class _AttributeArgumentRef(_ArgumentRef):
    """ A reference to an attribute on an argument. """

    def __init__(self, source: _ArgumentRef, attribute_name: str) -> None:
        self._source = source
        self._attribute_name = attribute_name

    @property
    def is_result(self) -> bool:
        return self._source.is_result

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        return getattr(self._source._get(arg_map), self._attribute_name)

    def __repr__(self) -> str:
        return f"{repr(self._source)}.{self._attribute_name}"


class _IndexArgumentRef(_ArgumentRef):
    """ A reference to an element in a list. """

    def __init__(self, source: _ArgumentRef, index: int) -> None:
        self._source = source
        self._index = index

    @property
    def is_result(self) -> bool:
        return self._source.is_result

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        return self._source._get(arg_map)[self._index]

    def __repr__(self) -> str:
        return f"{repr(self._source)}[{self._index}]"


@dataclass(frozen=True)
class _ParsedDimensionSpec:
    constant: Optional[int]
    variable_name: Optional[str]

    def __post_init__(self) -> None:
        assert (self.constant is None) != (
            self.variable_name is None
        ), "Argument must be either constant or variable."

    def __repr__(self) -> str:
        if self.constant is not None:
            return str(self.constant)
        else:
            assert self.variable_name is not None
            return self.variable_name


@dataclass(frozen=True)
class _ParsedShapeSpec:
    leading_dims_variable_name: Optional[str]
    trailing_dims: Tuple[_ParsedDimensionSpec, ...]

    def __repr__(self) -> str:
        dims = []
        if self.leading_dims_variable_name:
            dims.append(f"{self.leading_dims_variable_name}...")
        dims.extend(repr(dim) for dim in self.trailing_dims)
        return f"({', '.join(dims)})"


@dataclass(frozen=True)
class _ParsedArgumentSpec:
    argument_ref: _ArgumentRef
    shape: _ParsedShapeSpec

    def __repr__(self) -> str:
        return f"{self.argument_ref}: {self.shape}"


@experimental
def check_shapes(*specs: ArgumentSpec):
    """
    Decorator that checks the shapes of tensor arguments.

    This is compatible with both TensorFlow and NumPy.

    The specs passed to this decorator are (name, spec) tuples, where:
        name is a specification of the target to check the shape of.
            It can be the name of an argument.
            Or, it can be the special value "return", in which case the return value of the function
                is checked.
            Furthermore you can use dotted syntax (`argument.member1.member2`) to check members of
                objects.
            You can also use list lookup syntax (`argument[7]`) to access single members of tuples
                or lists.
        spec is a definition of the expected shape. spec is a sequence of one of:
            A constant integer. The corresponding dimension must have exactly this size.
            A variable name. The corresponding dimension can have any size, but must be the same
                everywhere that variable name is used.
            A variable name followed by ellipsis. This matches any number of dimensions. If used
                this must the first item in the spec sequence.

    Speed and interactions with `tf.function`:

    If you want to wrap your function in both `tf.function` and `check_shapes` it is recommended you
    put the `tf.function` outermost so that the shape checks are inside `tf.function`.
    Shape checks are performed while tracing graphs, but *not* compiled into the actual graphs.
    This is considered a feature as that means that `check_shapes` doesn't impact the execution
    speed of compiled functions. However, it also means that tensor dimensions of dynamic size are
    not verified in compiled mode.

    Example:

        @tf.function
        @check_shapes(
            ("features", ["batch_shape...", "n_features"]),
            ("weights", ["n_features"]),
            ("return", ["batch_shape..."]),
        )
        def linear_model(features: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
            ...
    """
    parsed_specs = _parse_specs(specs)

    # We create four groups of specs:
    # * Groups for checking before and after the function is called.
    # * Specs for printing in error message and specs for actually checking.
    pre_print_specs = [spec for spec in parsed_specs if not spec.argument_ref.is_result]
    post_print_specs = parsed_specs
    pre_check_specs = pre_print_specs
    post_check_specs = [spec for spec in parsed_specs if spec.argument_ref.is_result]

    def _check_shapes(func: _C) -> _C:
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                bound_arguments = signature.bind(*args, **kwargs)
            except TypeError:
                # TypeError is raised if *args and **kwargs don't actually match the arguments of
                # `func`. In that case we just call `func` normally, which will also result in an
                # error, but an error with the error message the user is used to.
                func(*args, **kwargs)
                raise AssertionError(
                    "The above line should fail so this line should never be reached."
                )
            bound_arguments.apply_defaults()
            arg_map = bound_arguments.arguments
            context: Dict[str, Union[int, List[Optional[int]]]] = {}
            _assert_shapes(func, pre_print_specs, pre_check_specs, arg_map, context)
            result = func(*args, **kwargs)
            arg_map[_RESULT_TOKEN] = result
            _assert_shapes(func, post_print_specs, post_check_specs, arg_map, context)
            return result

        wrapped.__check_shapes__ = _check_shapes  # type: ignore
        return cast(_C, wrapped)

    return _check_shapes


def _assert_shapes(
    func: _C,
    print_specs: Sequence[_ParsedArgumentSpec],
    check_specs: Sequence[_ParsedArgumentSpec],
    arg_map: Mapping[str, Any],
    context: Dict[str, Union[int, List[Optional[int]]]],
) -> None:
    def _assert(condition: bool):
        if not condition:
            raise ShapeMismatchError(func, print_specs, arg_map)

    for arg_spec in check_specs:
        actual_shape = arg_spec.argument_ref.get(func, arg_map).shape
        if isinstance(actual_shape, tf.TensorShape) and actual_shape.rank is None:
            continue

        actual = list(actual_shape)
        actual_len = len(actual)
        actual_i = 0

        expected = arg_spec.shape.trailing_dims
        expected_len = len(expected)
        expected_i = 0

        # Handle any leading variable-length check:
        if arg_spec.shape.leading_dims_variable_name is not None:
            expected_name = arg_spec.shape.leading_dims_variable_name
            _assert(expected_len <= actual_len)
            leading_dims_len = actual_len - expected_len
            actual_dims = actual[:leading_dims_len]
            actual_i += leading_dims_len

            expected_dims = context.get(expected_name)
            if expected_dims is None:
                expected_dims = cast(List[Optional[int]], leading_dims_len * [None])
                context[expected_name] = expected_dims

            assert isinstance(expected_dims, list)
            _assert(len(expected_dims) == len(actual_dims))
            for i, actual_dim in enumerate(actual_dims):
                if actual_dim is None:
                    continue
                if expected_dims[i] is None:
                    expected_dims[i] = actual_dim
                else:
                    _assert(expected_dims[i] == actual_dim)

        # Check that remaining number of dimensions is the same:
        _assert(expected_len - expected_i == actual_len - actual_i)

        # Handle normal single-dimension checks:
        while expected_i < expected_len:
            expected_item = expected[expected_i]
            actual_dim = actual[actual_i]

            if actual_dim is not None:
                if expected_item.constant is not None:
                    _assert(expected_item.constant == actual_dim)
                else:
                    assert expected_item.variable_name is not None
                    expected_dim = context.setdefault(expected_item.variable_name, actual_dim)
                    _assert(expected_dim == actual_dim)

            actual_i += 1
            expected_i += 1


def _parse_specs(raw_specs: Sequence[ArgumentSpec]) -> Sequence[_ParsedArgumentSpec]:
    argument_specs: List[_ParsedArgumentSpec] = []
    for argument_ref_str, raw_shape_spec in raw_specs:
        argument_ref = _parse_argument_ref(argument_ref_str)

        leading_dims_variable_name: Optional[str] = None
        trailing_dims: List[_ParsedDimensionSpec] = []
        is_first = True
        for i, raw_dimension_spec in enumerate(raw_shape_spec):
            if isinstance(raw_dimension_spec, int):
                trailing_dims.append(
                    _ParsedDimensionSpec(constant=raw_dimension_spec, variable_name=None)
                )
            else:
                assert isinstance(
                    raw_dimension_spec, str
                ), f"Invalid dimension specification type {type(raw_dimension_spec)}."
                match = _VARIABLE_RE.fullmatch(raw_dimension_spec)
                assert match, f"Invalid dimension specification {raw_dimension_spec}."
                if match[2] is None:
                    trailing_dims.append(
                        _ParsedDimensionSpec(constant=None, variable_name=match[1])
                    )
                else:
                    assert is_first, (
                        "Only the leading dimension can have variable length."
                        f" Found variable length for argument {argument_ref}, dimension {i}."
                    )
                    leading_dims_variable_name = match[1]
            is_first = False
        shape_spec = _ParsedShapeSpec(leading_dims_variable_name, tuple(trailing_dims))
        argument_specs.append(_ParsedArgumentSpec(argument_ref, shape_spec))
    return argument_specs


def _parse_argument_ref(argument_ref_str: str) -> _ArgumentRef:
    def _create_error_message() -> str:
        return f"Invalid argument reference: '{argument_ref_str}'."

    start = 0
    match: Optional[Match] = None

    def _consume(expression: Pattern) -> None:
        nonlocal start, match
        match = expression.match(argument_ref_str, start)
        if match:
            start = match.end()

    _consume(_ROOT_ARGUMENT_RE)
    assert match, _create_error_message()
    result: _ArgumentRef = _RootArgumentRef(match.group(0))
    while start < len(argument_ref_str):
        _consume(_ATTRIBUTE_ARGUMENT_RE)
        if match:
            result = _AttributeArgumentRef(result, match.group(1))
            continue
        _consume(_INDEX_ARGUMENT_RE)
        if match:
            result = _IndexArgumentRef(result, int(match.group(1)))
            continue
        assert False, _create_error_message()
    assert start == len(argument_ref_str), _create_error_message()
    return result


class ArgumentReferenceError(Exception):
    """ Error raised if the argument to check the shape of could not be resolved. """

    def __init__(self, func: _C, arg_map: Mapping[str, Any], arg_ref: _ArgumentRef) -> None:
        func_info = _FunctionDebugInfo.create(func)
        lines = [
            "Unable to resolve argument / missing argument.",
            f"    Function: {func_info.name}",
            f"    Declared: {func_info.path_and_line}",
            f"    Argument: {arg_ref}",
        ]

        super().__init__("\n".join(lines))

        self.func = func
        self.arg_map = arg_map
        self.arg_ref = arg_ref


class ShapeMismatchError(Exception):
    """ Error raised if a function is called with tensors of the wrong shape. """

    def __init__(
        self,
        func: _C,
        specs: Sequence[_ParsedArgumentSpec],
        arg_map: Mapping[str, Any],
    ) -> None:
        func_info = _FunctionDebugInfo.create(func)
        lines = [
            "Tensor shape mismatch in call to function.",
            f"    Function: {func_info.name}",
            f"    Declared: {func_info.path_and_line}",
        ]
        for spec in specs:
            actual_shape = spec.argument_ref.get(func, arg_map).shape
            if isinstance(actual_shape, tf.TensorShape) and actual_shape.rank is None:
                actual_str = "<Unknown>"
            else:
                actual_str = f"({', '.join(str(dim) for dim in actual_shape)})"
            lines.append(
                f"    Argument: {spec.argument_ref}, expected: {spec.shape}, actual: {actual_str}"
            )

        super().__init__("\n".join(lines))

        self.func = func
        self.specs = specs
        self.arg_map = arg_map


@dataclass
class _FunctionDebugInfo:
    """
    Information about a function, to print in error messages.
    """

    name: str
    path_and_line: str

    @staticmethod
    def create(func: _C) -> "_FunctionDebugInfo":
        name = func.__qualname__
        try:
            path = inspect.getsourcefile(func)
        except Exception:  # pragma: no cover
            path = "<unknown file>"
        try:
            _, line_int = inspect.getsourcelines(func)
            line = str(line_int)
        except Exception:  # pragma: no cover
            line = "<unknown lines>"
        path_and_line = f"{path}:{line}"

        return _FunctionDebugInfo(name=name, path_and_line=path_and_line)


@experimental
def inherit_check_shapes(func: _C) -> _C:
    """
    Decorator that inherits the `check_shapes` decoration from any overridden method in a
    super-class.

    Example:

        class SuperClass(ABC):
            @abstractmethod
            @check_shapes(
                ("a", ["batch...", 4]),
                ("return", ["batch...", 1]),
            )
            def f(self, a: tf.Tensor) -> tf.Tensor:
                ...

        class SubClass(SuperClass):
            @inherit_check_shapes
            def f(self, a: tf.Tensor) -> tf.Tensor:
                ...
    """
    return cast(_C, _InheritCheckShapes(func))


class _InheritCheckShapes:
    """
    Implementation of inherit_check_shapes.

    The __set_name__ hack is to get access to the class the method was declared on.
    See: https://stackoverflow.com/a/54316392 .
    """

    def __init__(self, func):
        self._func = func

    def __set_name__(self, owner: type, name: str) -> None:
        overridden_check_shapes: Optional[Callable[[_C], _C]] = None
        for parent in inspect.getmro(owner)[1:]:
            overridden_method = getattr(parent, name, None)
            if overridden_method is None:
                continue
            overridden_check_shapes = getattr(overridden_method, "__check_shapes__", None)
            if overridden_check_shapes is None:
                continue
            break

        assert overridden_check_shapes is not None, (
            f"@inherit_check_shapes did not find any overridden method of name '{name}'"
            f" on class '{owner.__name__}'."
        )

        self._func.class_name = owner.__name__
        wrapped = overridden_check_shapes(self._func)
        setattr(owner, name, wrapped)
