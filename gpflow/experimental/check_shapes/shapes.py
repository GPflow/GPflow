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
from functools import singledispatch
from typing import Any, Sequence, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .base_types import ActualShape


@singledispatch
def get_shape(shaped: Any) -> ActualShape:
    """
    Returns the shape of the given object.

    Returns `None` if the object has as shape, but it is unknown.

    Raises an exception if objects of that type do not have shapes.
    """
    raise NotImplementedError(f"Do not know how to get shape of object of type {type(shaped)}.")


@get_shape.register(bool)
@get_shape.register(int)
@get_shape.register(float)
def get_scalar_shape(shaped: Any) -> ActualShape:
    return ()


@get_shape.register(Sequence)
def get_sequence_shape(shaped: Sequence[Any]) -> ActualShape:
    if isinstance(shaped, str):
        raise NotImplementedError("Strings do not have shapes.")
    if len(shaped) == 0:
        # If the sequence doesn't have any elements we cannot use the first element to determine the
        # shape, and the shape is unknown.
        return None
    child_shape = get_shape(shaped[0])
    if child_shape is None:
        return None
    return (len(shaped),) + child_shape


@get_shape.register(np.ndarray)
def get_ndarray_shape(shaped: np.ndarray) -> ActualShape:
    return shaped.shape


@get_shape.register(tf.Tensor)
@get_shape.register(tf.Variable)
@get_shape.register(tfp.util.DeferredTensor)
def get_tensorflow_shape(
    shaped: Union[tf.Tensor, tf.Variable, tfp.util.DeferredTensor]
) -> ActualShape:
    shape = shaped.shape
    if not shape:
        return None
    return tuple(shape)
