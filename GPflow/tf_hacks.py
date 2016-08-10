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
A collection of hacks for tensorflow.

Hopefully we can remove these as the library matures
"""

import os
import tensorflow as tf


def eye(N):
    return tf.diag(tf.ones(tf.pack([N, ]), dtype='float64'))


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
