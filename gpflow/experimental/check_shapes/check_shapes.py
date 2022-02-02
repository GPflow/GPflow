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
from .specs import ArgumentSpec, ParsedArgumentSpec, parse_specs


@experimental
def check_shapes(*specs: ArgumentSpec) -> Callable[[C], C]:
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
    parsed_specs = parse_specs(specs)

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
