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
Decorator for checking the shapes of function using tf Tensors.
"""
import inspect
from functools import update_wrapper
from typing import Any, Callable, Sequence, cast

import tensorflow as tf

from ..utils import experimental
from .accessors import set_check_shapes
from .argument_ref import RESULT_TOKEN
from .base_types import C
from .checker import ShapeChecker
from .checker_context import set_shape_checker
from .config import get_enable_check_shapes
from .error_contexts import (
    ConditionContext,
    FunctionCallContext,
    FunctionDefinitionContext,
    NoteContext,
    ParallelContext,
    StackContext,
)
from .parser import parse_and_rewrite_docstring, parse_function_spec
from .specs import ParsedArgumentSpec


def null_check_shapes(func: C) -> C:
    """
    Annotates the given function so that it looks like it has shape checks, but without actually
    checking anything.

    This is necessary not to break ``@inherit_check_shapes`` when shape checking is disabled.
    """
    set_check_shapes(func, null_check_shapes)
    return func


@experimental
def check_shapes(*specs: str, tf_decorator: bool = False) -> Callable[[C], C]:
    """
    Decorator that checks the shapes of tensor arguments.

    Example:

    .. literalinclude:: /examples/test_check_shapes_examples.py
       :start-after: [basic]
       :end-before: [basic]
       :dedent:

    :param specs: Specification of arguments to check. See: `Check specification`_.
    :param tf_decorator: Whether to wrap the shape check with
        ``tf.compat.v1.flags.tf_decorator.make_decorator``.
        Setting this `True` seems to solve some problems, particularly related to Keras models,
        but create some other problems, particularly related to branching on tensors.
    """
    if not get_enable_check_shapes():
        return null_check_shapes

    unbound_error_context = FunctionCallContext(check_shapes)

    func_spec = parse_function_spec(specs, unbound_error_context)

    pre_specs = [spec for spec in func_spec.arguments if not spec.argument_ref.is_result]
    post_specs = [spec for spec in func_spec.arguments if spec.argument_ref.is_result]
    note_specs = func_spec.notes

    def _check_shapes(func: C) -> C:
        bound_error_context = FunctionDefinitionContext(func)
        signature = inspect.signature(func)

        def wrapped_function(*args: Any, **kwargs: Any) -> Any:
            if not get_enable_check_shapes():
                return func(*args, **kwargs)

            try:
                bound_arguments = signature.bind(*args, **kwargs)
            except TypeError as e:
                # TypeError is raised if *args and **kwargs don't actually match the arguments of
                # `func`. In that case we just call `func` normally, which will also result in an
                # error, but an error with the error message the user is used to.
                try:
                    func(*args, **kwargs)
                except TypeError as e2:
                    raise TypeError(
                        "Error calling wrapped function (see above error)."
                        " If you believe your parameters actually are correct, the error can"
                        " sometimes be fixed by setting `tf_decorator=True` on your `@check_shapes`"
                        " decorator."
                    ) from e2
                raise AssertionError(
                    "The above line should fail so this line should never be reached."
                ) from e
            bound_arguments.apply_defaults()
            arg_map = bound_arguments.arguments

            checker = ShapeChecker()
            for note_spec in note_specs:
                checker.add_context(StackContext(bound_error_context, NoteContext(note_spec)))

            def _check_specs(specs: Sequence[ParsedArgumentSpec]) -> None:
                processed_specs = []

                for arg_spec in specs:
                    for arg_value, relative_arg_context in arg_spec.argument_ref.get(
                        arg_map, bound_error_context
                    ):
                        arg_context = StackContext(bound_error_context, relative_arg_context)

                        if arg_spec.condition is not None:
                            condition, condition_context = arg_spec.condition.get(
                                arg_map,
                                StackContext(arg_context, ConditionContext(arg_spec.condition)),
                            )
                            if not condition:
                                continue
                            arg_context = StackContext(
                                bound_error_context,
                                ParallelContext(
                                    (
                                        StackContext(
                                            relative_arg_context,
                                            StackContext(
                                                ConditionContext(arg_spec.condition),
                                                condition_context,
                                            ),
                                        ),
                                    )
                                ),
                            )

                        processed_specs.append((arg_value, arg_spec.tensor, arg_context))

                checker.check_shapes(processed_specs)

            _check_specs(pre_specs)

            with set_shape_checker(checker):
                result = func(*args, **kwargs)
            arg_map[RESULT_TOKEN] = result

            _check_specs(post_specs)

            return result

        # Work-around for TensorFlow saved_model expecting methods to have a `self` argument:
        if "self" in signature.parameters:

            def wrapped_method(self: Any, *args: Any, **kwargs: Any) -> Any:
                return wrapped_function(self, *args, **kwargs)

            wrapped = wrapped_method
        else:
            wrapped = wrapped_function

        # Make TensorFlow understand our decoration:
        if tf_decorator:
            tf.compat.v1.flags.tf_decorator.make_decorator(func, wrapped)

        update_wrapper(wrapped, func)
        set_check_shapes(wrapped, _check_shapes)
        wrapped.__doc__ = parse_and_rewrite_docstring(func.__doc__, func_spec, bound_error_context)
        return cast(C, wrapped)

    return _check_shapes
