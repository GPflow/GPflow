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
import warnings


from . import settings


logger = settings.logger()


def gaussian(x, mu, var):
    return -0.5 * (np.log(2 * np.pi) + tf.log(var) + tf.square(mu-x) / var)


def lognormal(x, mu, var):
    lnx = tf.log(x)
    return gaussian(lnx, mu, var) - lnx


def bernoulli(x, p):
    return tf.log(tf.where(tf.equal(x, 1), p, 1-p))


def poisson(x, lam):
    return x * tf.log(lam) - lam - tf.lgamma(x + 1.)


def exponential(x, scale):
    return - x/scale - tf.log(scale)


def gamma(x, shape, scale):
    return -shape * tf.log(scale) - tf.lgamma(shape) \
        + (shape - 1.) * tf.log(x) - x / scale


def student_t(x, mean, scale, df):
    df = tf.cast(df, settings.float_type)
    const = tf.lgamma((df + 1.) * 0.5) - tf.lgamma(df * 0.5) \
        - 0.5 * (tf.log(tf.square(scale)) + tf.log(df) + np.log(np.pi))
    const = tf.cast(const, settings.float_type)
    return const - 0.5 * (df + 1.) * \
        tf.log(1. + (1. / df) * (tf.square((x - mean) / scale)))


def beta(x, alpha, beta):
    # need to clip x, since log of 0 is nan...
    x = tf.clip_by_value(x, 1e-6, 1-1e-6)
    return (alpha - 1.) * tf.log(x) + (beta - 1.) * tf.log(1. - x) \
        + tf.lgamma(alpha + beta)\
        - tf.lgamma(alpha)\
        - tf.lgamma(beta)


def laplace(x, mu, sigma):
    return - tf.abs(mu - x) / sigma - tf.log(2. * sigma)


def multivariate_normal(x, mu, L):
    """
    Computes the log-density of a multivariate normal.
    :param x  : Dx1 or DxN sample(s) for which we want the density
    :param mu : Dx1 or DxN mean(s) of the normal distribution
    :param L  : DxD Cholesky decomposition of the covariance matrix
    :return p : (1,) or (N,) vector of log densities for each of the N x's and/or mu's

    x and mu are either vectors or matrices. If both are vectors (N,1):
    p[0] = log pdf(x) where x ~ N(mu, LL^T)
    If at least one is a matrix, we assume independence over the *columns*:
    the number of rows must match the size of L. Broadcasting behaviour:
    p[n] = log pdf of:
    x[n] ~ N(mu, LL^T) or x ~ N(mu[n], LL^T) or x[n] ~ N(mu[n], LL^T)
    """
    if x.shape.ndims is None:
        logger.warn('Shape of x must be 2D at computation.')
    elif x.shape.ndims != 2:
        raise ValueError('Shape of x must be 2D.')
    if mu.shape.ndims is None:
        logger.warn('Shape of mu may be unknown or not 2D.')
    elif mu.shape.ndims != 2:
        raise ValueError('Shape of mu must be 2D.')

    d = x - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_dims = tf.cast(tf.shape(d)[0], L.dtype)
    p = - 0.5 * tf.reduce_sum(tf.square(alpha), 0)
    p -= 0.5 * num_dims * np.log(2 * np.pi)
    p -= tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
    return p
