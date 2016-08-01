"""
A collection of hacks for tensorflow.

Hopefully we can remove these as the library matures
"""

import os
import tensorflow as tf


def eye(N):
    return tf.diag(tf.ones(tf.pack([N, ]), dtype='float64'))

_vec_to_tri_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'tfops', 'vec_to_tri.so'))
vec_to_tri = _vec_to_tri_module.vec_to_tri
