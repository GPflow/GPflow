import os
import tensorflow as tf

_vec_to_tri_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'vec_to_tri.so'))
vec_to_tri = _vec_to_tri_module.vec_to_tri
