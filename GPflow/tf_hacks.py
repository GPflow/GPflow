"""
A collection of hacks for tensorflow.

Hopefully we can remove these as the library matures
"""

import tensorflow as tf


def eye(N):
    return tf.diag(tf.ones(tf.pack([N, ]), dtype='float64'))


def diag_1stdim(X):
    """
    X is tensor.
    
    Make an diagonalized tensor along the 1st dimension.
    
    e.g.
    X = [[[x000, x001, x002], [x010, x011, x012], [x020, x021, x022]], 
        [[[x100, x101, x102], [x110, x111, x112], [x120, x121, x122]]] 

    tf.shape(X) => [2, 3, 3]
    
    tf.shape(diag_1st_dim(X)) => [2,2,3,3]
    
    diag_1st_dim(X) = 
        [[[[x000, x001, x002], [x010, x011, x012], [x020, x021, x022]], [[0,0,0],[0,0,0],[0,0,0]]]
          [[0,0,0],[0,0,0],[0,0,0]], [[[x100, x101, x102], [x110, x111, x112], [x120, x121, x122]]]]
    
    Trick used in this method:
    1. Consider 1-dimensional two-element tensor [x1, x2]
    2. Expand a dimension, [x1, x2] -> [[x1], [x2]]
                    shape [2] -> [2,1]
    3. Pad zeros along the second dimension, [[x1,0,0],[x2,0,0]]
       The shape will be [2,1] -> [2,3]
    4. Reshape [2,3] -> [3,2]
       The tensor will be [[x1,0,0],[x2,0,0]] -> [[x1,0],[0,x2],[0,0]]
    5. Remove the last element along in the first dimension
       [[x1,0],[0,x2],[0,0]] -> [[x1,0],[0,x2]]
       Diagonalized!!
    """
    # Example, tf.shape(X) -> [2,3,3]
    X_expand = tf.expand_dims(X, dim=1)
    # tf.shape(X_expand) -> [2,1,3,3]
    rank = tf.rank(X_expand)
    shape = tf.shape(X_expand)

    # atomic vector with only index1 is 1 otherwize zero. 
    # atomic_0 -> [1, 0, 0, 0]
    atomic_0 = tf.concat(0, [[1], [0], tf.zeros([rank-2], dtype=tf.int32)])
    # atomic vector with only index1 is 1 otherwize zero
    # atomic_1 -> [0, 1, 0, 0]
    atomic_1 = tf.concat(0, [[0], [1], tf.zeros([rank-2], dtype=tf.int32)])
    
    # pad zeros along dimension-1, to satisfy the tf.shape(X_pad)[0] + 1 == tf.shape(X_pad)[1]
    padding = tf.transpose(tf.pack([tf.zeros_like(atomic_1), atomic_1 * shape[0]]))
    # tf.shape(X_pad) -> [2,3,3,3]
    X_pad = tf.pad(X_expand, padding)
    # reshape X_pad so that tf.shape(X_pad[1]) == tf.shape(X_reshape[0]) and
    #                       tf.shape(X_pad[0]) == tf.shape(X_reshape[1]) 
    X_reshape = tf.reshape(X_pad, tf.shape(X_pad)-atomic_1+atomic_0)

    # remove the last element in dimension 1.
    return tf.slice(X_reshape, tf.zeros_like(tf.shape(X_reshape)), tf.shape(X_reshape)-atomic_0)
