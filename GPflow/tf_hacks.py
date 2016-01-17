"""
A collection of hacks for tensorflow.

hopefully we can remove these as the library matures
"""

import tensorflow as tf

def eye(N):
    return tf.diag(tf.ones(tf.pack([N,]), dtype='float64') )

