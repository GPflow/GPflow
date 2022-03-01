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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast

import tensorflow as tf

from ..utils import experimental
from .argument_ref import RESULT_TOKEN
from .base_types import C
from .errors import ShapeMismatchError
from .parser import parse_and_rewrite_docstring, parse_argument_spec
from .specs import ParsedArgumentSpec


@experimental
def check_shapes(*specs: str) -> Callable[[C], C]:
    """
    Decorator that checks the shapes of tensor arguments.

    See: `check_shapes`_.

    :param spec_strs: Specification of arguments to check. See: `Argument specification`_.
    """
    parsed_specs = tuple(parse_argument_spec(spec) for spec in specs)

    # We create four groups of specs:
    # * Groups for checking before and after the function is called.
    # * Specs for printing in error message and specs for actually checking.
    pre_print_specs = [spec for spec in parsed_specs if not spec.argument_ref.is_result]
    post_print_specs = parsed_specs
    pre_check_specs = pre_print_specs
    post_check_specs = [spec for spec in parsed_specs if spec.argument_ref.is_result]

    def _check_shapes(func: C) -> C:
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
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
            arg_map[RESULT_TOKEN] = result
            _assert_shapes(func, post_print_specs, post_check_specs, arg_map, context)
            return result

        wrapped.__check_shapes__ = _check_shapes  # type: ignore
        wrapped.__doc__ = parse_and_rewrite_docstring(wrapped.__doc__, parsed_specs)
        return cast(C, wrapped)

    return _check_shapes


def _assert_shapes(
    func: C,
    print_specs: Sequence[ParsedArgumentSpec],
    check_specs: Sequence[ParsedArgumentSpec],
    arg_map: Mapping[str, Any],
    context: Dict[str, Union[int, List[Optional[int]]]],
) -> None:
    def _assert(condition: bool) -> None:
        if not condition:
            raise ShapeMismatchError(func, print_specs, arg_map)

    for arg_spec in check_specs:
        actual_shape = arg_spec.argument_ref.get(func, arg_map).shape
        if isinstance(actual_shape, tf.TensorShape) and actual_shape.rank is None:
            continue

        actual = list(actual_shape)
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
                assert (
                    dim_spec.variable_name is not None
                ), "All variable-rank ParsedDimensionSpec must be bound to a variable."

                expected_name = dim_spec.variable_name
                variable_rank_len = actual_len - (expected_len - n_variable_rank)
                actual_dims = actual[actual_i : actual_i + variable_rank_len]
                actual_i += variable_rank_len

                expected_dims = context.get(expected_name)
                if expected_dims is None:
                    expected_dims = cast(List[Optional[int]], variable_rank_len * [None])
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

            else:
                actual_dim = actual[actual_i]
                if actual_dim is not None:
                    if dim_spec.constant is not None:
                        _assert(dim_spec.constant == actual_dim)
                    else:
                        assert dim_spec.variable_name is not None
                        expected_dim = context.setdefault(dim_spec.variable_name, actual_dim)
                        _assert(expected_dim == actual_dim)
                actual_i += 1
