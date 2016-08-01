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
    i = tri_to_vec(grad[0, :, :])
    return [tf.reshape(i, [1, tf.shape(i)[0]])]


# TODO: Finish registering the shape. Was unsure how to handle incomplete shape information in the input.
# @tf.RegisterShape("VecToTri")
# def _vec_to_tri_shape(op):
#     in_shape = op.inputs[0].get_shape()
#     k = int((in_shape[1] * 8 + 1) ** 0.5 / 2.0 - 0.5);
#     shape = tf.TensorShape({in_shape[0], k, k})
#     return [shape]
