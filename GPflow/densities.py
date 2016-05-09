import tensorflow as tf
import numpy as np

def gaussian(x, mu, var):
    return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(var) - 0.5 * tf.square(mu-x)/var

def bernoulli(p, y):
    return tf.log(tf.select(tf.equal(y, 1), p, 1-p))

def poisson(lamb, y):
    return y * tf.log(lamb) - lamb - tf.lgamma(y + 1.)

def exponential(lamb, y):
    return - y/lamb - tf.log(lamb)

def gamma(shape, scale, x):
    return -shape * tf.log(scale) - tf.lgamma(shape) + (shape - 1.) * tf.log(x) - x / scale

def student_t(x, mean, scale, deg_free):
    const = tf.lgamma(tf.cast((deg_free + 1.) * 0.5, tf.float64))\
          - tf.lgamma(tf.cast(deg_free * 0.5, tf.float64))\
          - 0.5*(tf.log(tf.square(scale)) + tf.cast(tf.log(deg_free), tf.float64) + np.log(np.pi))
    const = tf.cast(const, tf.float64)
    return const - 0.5*(deg_free + 1.)*tf.log(1. + (1./deg_free)*(tf.square((x-mean)/scale)))

def beta(alpha, beta, y):
    #need to clip y, since log of 0 is nan...
    y = tf.clip_by_value(y, 1e-6, 1-1e-6)
    return (alpha - 1.) * tf.log(y) + (beta - 1.) * tf.log(1. - y) \
            + tf.lgamma(alpha + beta)\
            - tf.lgamma(alpha)\
            - tf.lgamma(beta)

           



def multivariate_normal(x, mu, L):
    """
    L is the Cholesky decomposition of the covaraince.

    x and mu are either vectors (ndim=1) or matrices. in the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_col = 1 if tf.rank(x)==1 else tf.shape(x)[1]
    #TODO: this call to get_diag relies on x being a numpy object (ie. having a shape)
    ret =  - 0.5 * tf.cast(tf.size(x), tf.float64) * np.log(2 * np.pi) 
    ret += - tf.cast(num_col, tf.float64) * tf.reduce_sum(tf.log(tf.diag_part(L)))
    ret += - 0.5 * tf.reduce_sum(tf.square(alpha))
    return ret

