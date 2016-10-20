# Copyright 2016 James Hensman, alexggmatthews
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
A collection of wrappers and extensions for tensorflow.
"""

import os
import tensorflow as tf
from ._settings import settings


def eye(N):
    """
    An identitiy matrix
    """
    return tf.diag(tf.ones(tf.pack([N, ]), dtype=settings.dtypes.float_type))


_custom_op_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'tfops', 'matpackops.so'))
vec_to_tri = _custom_op_module.vec_to_tri
tri_to_vec = _custom_op_module.tri_to_vec


@tf.python.framework.ops.RegisterGradient("VecToTri")
def _vec_to_tri_grad(op, grad):
    return [tri_to_vec(grad)]


@tf.RegisterShape("VecToTri")
def _vec_to_tri_shape(op):
    in_shape = op.inputs[0].get_shape().with_rank(2)
    M = in_shape[1].value
    if M is None:
        k = None
    else:
        k = int((M * 8 + 1) ** 0.5 / 2.0 - 0.5)
    shape = tf.TensorShape([in_shape[0], k, k])
    return [shape]

_custom_op_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'tfops', 'rowdeleteops.so'))
remove_row_elements = _custom_op_module.remove_row_elements
remove_row_elements_grad = _custom_op_module.remove_row_elements_grad


@tf.python.framework.ops.RegisterGradient("RemoveRowElements")
def _remove_row_elements_grad(op, gradmat, gradvec):
    index = op.inputs[1]
    return [remove_row_elements_grad(gradmat, gradvec, index), tf.zeros_like(index)]


@tf.RegisterShape("RemoveRowElements")
def _remove_row_elements_shape(op):
    in_shape = op.inputs[0].get_shape().with_rank(2)
    shape1 = tf.TensorShape([in_shape[0], in_shape[1] - 1])
    shape2 = tf.TensorShape([in_shape[0]])
    return [shape1, shape2]
