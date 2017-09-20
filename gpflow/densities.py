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


import tensorflow as tf
import numpy as np
from ._settings import settings
float_type = settings.dtypes.float_type


def gaussian(x, mu, var):
    return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(var)\
        - 0.5 * tf.square(mu-x)/var


def lognormal(x, mu, var):
    lnx = tf.log(x)
    return gaussian(lnx, mu, var) - lnx


def bernoulli(p, y):
    return tf.log(tf.where(tf.equal(y, 1), p, 1-p))


def poisson(lamb, y):
    return y * tf.log(lamb) - lamb - tf.lgamma(y + 1.)


def exponential(lamb, y):
    return - y/lamb - tf.log(lamb)


def gamma(shape, scale, x):
    return -shape * tf.log(scale) - tf.lgamma(shape)\
        + (shape - 1.) * tf.log(x) - x / scale


def student_t(x, mean, scale, deg_free):
    const = tf.lgamma(tf.cast((deg_free + 1.) * 0.5, float_type))\
        - tf.lgamma(tf.cast(deg_free * 0.5, float_type))\
        - 0.5*(tf.log(tf.square(scale)) + tf.cast(tf.log(deg_free), float_type)
               + np.log(np.pi))
    const = tf.cast(const, float_type)
    return const - 0.5*(deg_free + 1.) * \
        tf.log(1. + (1. / deg_free) * (tf.square((x - mean) / scale)))


def beta(alpha, beta, y):
    # need to clip y, since log of 0 is nan...
    y = tf.clip_by_value(y, 1e-6, 1-1e-6)
    return (alpha - 1.) * tf.log(y) + (beta - 1.) * tf.log(1. - y) \
        + tf.lgamma(alpha + beta)\
        - tf.lgamma(alpha)\
        - tf.lgamma(beta)


def laplace(mu, sigma, y):
    return - tf.abs(mu - y) / sigma - tf.log(2. * sigma)


def multivariate_normal(x, mu, L):
    """
    L is the Cholesky decomposition of the covariance.

    x and mu are either vectors (ndim=1) or matrices. In the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_col = 1 if tf.rank(x) == 1 else tf.shape(x)[1]
    num_col = tf.cast(num_col, float_type)
    num_dims = tf.cast(tf.shape(x)[0], float_type)
    ret = - 0.5 * num_dims * num_col * np.log(2 * np.pi)
    ret += - num_col * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
    ret += - 0.5 * tf.reduce_sum(tf.square(alpha))
    return ret
