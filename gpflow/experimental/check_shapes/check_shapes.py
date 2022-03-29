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
from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

from ..utils import experimental
from .accessors import set_check_shapes
from .argument_ref import RESULT_TOKEN
from .base_types import C, Shape
from .config import get_enable_check_shapes
from .error_contexts import (
    ArgumentContext,
    ErrorContext,
    FunctionCallContext,
    FunctionDefinitionContext,
    ParallelContext,
    ShapeContext,
    StackContext,
)
from .exceptions import ShapeMismatchError
from .parser import parse_and_rewrite_docstring, parse_argument_spec
from .shapes import get_shape
from .specs import ParsedArgumentSpec


def null_check_shapes(func: C) -> C:
    """
    Annotates the given function so that it looks like it has shape checks, but without actually
    checking anything.

    This is necessary not to break `@inherit_check_shapes` when shape checking is disabled.
    """
    set_check_shapes(func, null_check_shapes)
    return func


@experimental
def check_shapes(*specs: str) -> Callable[[C], C]:
    """
    Decorator that checks the shapes of tensor arguments.

    See: `check_shapes`_.

    :param spec_strs: Specification of arguments to check. See: `Argument specification`_.
    """
    if not get_enable_check_shapes():
        return null_check_shapes

    unbound_error_context = FunctionCallContext(check_shapes)

    parsed_specs = tuple(
        parse_argument_spec(spec, StackContext(unbound_error_context, ArgumentContext(i)))
        for i, spec in enumerate(specs)
    )

    pre_specs = [spec for spec in parsed_specs if not spec.argument_ref.is_result]
    post_specs = [spec for spec in parsed_specs if spec.argument_ref.is_result]

    def _check_shapes(func: C) -> C:
        bound_error_context = FunctionDefinitionContext(func)
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not get_enable_check_shapes():
                return func(*args, **kwargs)

            try:
                bound_arguments = signature.bind(*args, **kwargs)
            except TypeError as e:
                # TypeError is raised if *args and **kwargs don't actually match the arguments of
                # `func`. In that case we just call `func` normally, which will also result in an
                # error, but an error with the error message the user is used to.
                func(*args, **kwargs)
                raise AssertionError(
                    "The above line should fail so this line should never be reached."
                ) from e
            bound_arguments.apply_defaults()
            arg_map = bound_arguments.arguments
            shape_context: List[Tuple[ParsedArgumentSpec, Shape]] = []
            dim_context: Dict[str, Union[int, List[Optional[int]]]] = {}
            _assert_shapes(pre_specs, arg_map, shape_context, dim_context, bound_error_context)
            result = func(*args, **kwargs)
            arg_map[RESULT_TOKEN] = result
            _assert_shapes(post_specs, arg_map, shape_context, dim_context, bound_error_context)
            return result

        set_check_shapes(wrapped, _check_shapes)
        wrapped.__doc__ = parse_and_rewrite_docstring(
            wrapped.__doc__, parsed_specs, bound_error_context
        )
        return cast(C, wrapped)

    return _check_shapes


def _assert_shapes(
    specs: Sequence[ParsedArgumentSpec],
    arg_map: Mapping[str, Any],
    shape_context: List[Tuple[ParsedArgumentSpec, Shape]],
    dim_context: Dict[str, Union[int, List[Optional[int]]]],
    error_context: ErrorContext,
) -> None:
    def _assert(condition: bool) -> None:
        if not condition:
            raise ShapeMismatchError(
                StackContext(
                    error_context,
                    ParallelContext(
                        [
                            StackContext(
                                spec.argument_ref.error_context, ShapeContext(spec.shape, actual)
                            )
                            for spec, actual in shape_context
                        ]
                    ),
                )
            )

    new_shape_context = []
    for arg_spec in specs:
        arg_value = arg_spec.argument_ref.get(arg_map, error_context)
        actual: Shape
        if arg_value is None:
            actual = None
        else:
            actual = get_shape(
                arg_value, StackContext(error_context, arg_spec.argument_ref.error_context)
            )
        shape_context.append((arg_spec, actual))
        if actual is not None:
            new_shape_context.append((arg_spec, actual))

    for arg_spec, actual in new_shape_context:
        actual_len = len(actual)
        actual_i = 0

        expected = arg_spec.shape.dims
        expected_len = len(expected)

        n_variable_rank = sum(dim_spec.variable_rank for dim_spec in expected)
        assert n_variable_rank <= 1, "At most one variable-rank ParsedDimensionSpec allowed."
        if n_variable_rank == 0:
            _assert(expected_len == actual_len)
        else:
            _assert(expected_len - n_variable_rank <= actual_len)

        for dim_spec in expected:

            if dim_spec.variable_rank:
                variable_rank_len = actual_len - (expected_len - n_variable_rank)
                actual_dims = actual[actual_i : actual_i + variable_rank_len]
                actual_i += variable_rank_len

                expected_name = dim_spec.variable_name
                if expected_name is None:
                    # Anonymous dimension spec - we don't care about the actual values.
                    continue

                expected_dims = dim_context.get(expected_name)
                if expected_dims is None:
                    expected_dims = cast(List[Optional[int]], variable_rank_len * [None])
                    dim_context[expected_name] = expected_dims

                assert isinstance(expected_dims, list)

                _assert(len(expected_dims) == len(actual_dims))
                for i, actual_dim in enumerate(actual_dims):
                    if actual_dim is None:
                        continue
                    if expected_dims[i] is None:
                        expected_dims[i] = actual_dim
                    else:
                        _assert(expected_dims[i] == actual_dim)

            else:
                actual_dim = actual[actual_i]
                if actual_dim is not None:
                    if dim_spec.constant is not None:
                        _assert(dim_spec.constant == actual_dim)
                    elif dim_spec.variable_name is not None:
                        expected_dim = dim_context.setdefault(dim_spec.variable_name, actual_dim)
                        _assert(expected_dim == actual_dim)
                    else:
                        pass  # Anonymous dimension - we don't care about the actual value.
                actual_i += 1
