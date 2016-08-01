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

# def _vec_to_tri