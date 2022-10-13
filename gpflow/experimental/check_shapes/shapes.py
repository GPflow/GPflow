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
Code for extracting shapes from object.
"""
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Tuple, Type

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .base_types import Shape
from .error_contexts import ErrorContext, IndexContext, ObjectTypeContext, StackContext
from .exceptions import NoShapeError

if TYPE_CHECKING:  # pragma: no cover
    # Avoid cyclic imports:
    from ...base import AnyNDArray
else:
    AnyNDArray = Any
GetShape = Callable[[Any, ErrorContext], Shape]


_GET_SHAPES: Dict[Type[Any], GetShape] = {}


def register_get_shape(shape_type: Type[Any]) -> Callable[[GetShape], GetShape]:
    """
    Register a function for extracting the shape from a given type of objects.

    Example:

    .. literalinclude:: /examples/test_check_shapes_examples.py
       :start-after: [custom_type]
       :end-before: [custom_type]
       :dedent:

    See also :func:`get_shape`.

    :param shape_type: Type of objects to extract shapes from.
    """
    # Yes, what's happening here looks extremely much like `functools.singledispatch`;
    # however we cannot actually use `functools.singledispatch`, because it uses a for/else
    # statement, which TensorFlow doesn't know how to compile...

    def _register(getter: GetShape) -> GetShape:
        _GET_SHAPES[shape_type] = getter
        return getter

    return _register


def get_shape(shaped: Any, context: ErrorContext) -> Shape:
    """
    Returns the shape of the given object.

    See also :func:`register_get_shape`.

    :param shaped: The objects whose shape to extract.
    :param context: Context we are getting the shape in, for improved error messages.
    :returns: The shape of ``shaped``, or ``None`` if the shape exists, but is unknown.
    :raises NoShapeError: If objects of this type does not have shapes.
    """
    for t in inspect.getmro(shaped.__class__):
        getter = _GET_SHAPES.get(t)
        if getter is not None:
            return getter(shaped, context)

    raise NoShapeError(StackContext(context, ObjectTypeContext(shaped)))


@register_get_shape(bool)
@register_get_shape(int)
@register_get_shape(float)
@register_get_shape(str)
def get_scalar_shape(shaped: Any, context: ErrorContext) -> Shape:
    return ()


@register_get_shape(list)
@register_get_shape(tuple)
def get_sequence_shape(shaped: Sequence[Any], context: ErrorContext) -> Shape:
    if len(shaped) == 0:
        # If the sequence doesn't have any elements we cannot use the first element to determine the
        # shape, and the shape is unknown.
        return None
    child_shape = get_shape(shaped[0], StackContext(context, IndexContext(0)))
    if child_shape is None:
        return None
    return (len(shaped),) + child_shape


@register_get_shape(np.ndarray)
def get_ndarray_shape(shaped: AnyNDArray, context: ErrorContext) -> Shape:
    result: Tuple[int, ...] = shaped.shape
    return result


@register_get_shape(tf.Tensor)
@register_get_shape(tf.Variable)
@register_get_shape(tfp.util.DeferredTensor)
def get_tensorflow_shape(shaped: Any, context: ErrorContext) -> Shape:
    shape = shaped.shape
    if not shape:
        return None
    return tuple(shape)


@register_get_shape(tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible)
def get_tensor_coercible_shape(shaped: Any, context: ErrorContext) -> Shape:
    # This one is unpleasant. Sometimes the `shape` is a `TensorShape`, but sometimes it's a
    # function that returns a `TensorShape`. The version of TensorFlow probability seems to have
    # something to do with it, but it also seems to be more complicated...
    shape = shaped.shape
    if not shape:
        return None
    if not isinstance(shape, tf.TensorShape):
        shape = shape()
        if not shape:
            return None
    return tuple(shape)
