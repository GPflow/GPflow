import tensorflow as tf
import numpy as np

float_type = tf.float32

# work out the np dty[e
if float_type is tf.float64:
    np_float_type = np.float64
elif float_type is tf.float32:
    np_float_type = np.float32
