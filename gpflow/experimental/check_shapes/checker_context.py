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
Utilities for accessing the ShapeChecker from a wrapping `check_shapes` decorator.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from .checker import S, ShapeChecker, TensorSpecLike
from .config import get_enable_check_shapes
from .error_contexts import ErrorContext, FunctionCallContext

_shape_checker: ContextVar[ShapeChecker] = ContextVar("_shape_checker")


@contextmanager
def set_shape_checker(checker: ShapeChecker) -> Iterator[None]:
    """
    Sets the current shape checker in the context.
    """
    token = _shape_checker.set(checker)
    try:
        yield
    finally:
        _shape_checker.reset(token)


def get_shape_checker() -> ShapeChecker:
    """
    Get the :class:`ShapeChecker` from the wrapping :func:`check_shapes` decorator.

    Behaviour is undefined if you call this from a function that is not directly wrapped in
    :func:`check_shapes` or :func:`inherit_check_shapes`.
    """
    return _shape_checker.get()


def check_shape(
    shaped: S, tensor_spec: TensorSpecLike, context: Optional[ErrorContext] = None
) -> S:
    """
    Raise an error if a tensor has the wrong shape.

    This uses the :class:`ShapeChecker` from the wrapping :func:`check_shapes` decorator. Behaviour
    is undefined if you call this from a function that is not directly wrapped in
    :func:`check_shapes` or :func:`inherit_check_shapes`.

    Example:

    .. literalinclude:: /examples/test_check_shapes_examples.py
       :start-after: [intermediate_results]
       :end-before: [intermediate_results]
       :dedent:

    :param shaped: The object whose shape to check.
    :param tensor_spec: Specification to check the tensor against. See: `Shape specification`_.
    :param context: Information about where ``shaped`` is coming from, for improved error
        messages.
    :returns: ``shaped``, for convenience.
    """
    if not get_enable_check_shapes():
        return shaped

    if context is None:
        context = FunctionCallContext(check_shape).precompute()

    return get_shape_checker().check_shape(shaped, tensor_spec, context)
