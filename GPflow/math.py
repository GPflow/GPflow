import tensorflow as tf
from ._settings import settings
from . import tf_wraps

def jitteredCholesky(A):
	# Returns Cholesky decomposition 
	# of square matrix A after adding 
	# a small amount of the identity  
	# to improve numerical stability.
	epsilon = tf.reduce_max(A) * settings.numerics.jitter_level  
	identity = tf_wraps.eye(tf.shape(A)[0])
	jitterA = A + epsilon * identity
	L = tf.cholesky(jitterA)
	return L
