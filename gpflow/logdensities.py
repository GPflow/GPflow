# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf

from .base import TensorType
from .experimental.check_shapes import check_shapes
from .utilities import to_default_float


@check_shapes(
    "x: [broadcast shape...]",
    "mu: [broadcast shape...]",
    "var: [broadcast shape...]",
    "return: [shape...]",
)
def gaussian(x: TensorType, mu: TensorType, var: TensorType) -> tf.Tensor:
    return -0.5 * (np.log(2 * np.pi) + tf.math.log(var) + tf.square(mu - x) / var)


@check_shapes(
    "x: [broadcast shape...]",
    "mu: [broadcast shape...]",
    "var: [broadcast shape...]",
    "return: [shape...]",
)
def lognormal(x: TensorType, mu: TensorType, var: TensorType) -> tf.Tensor:
    lnx = tf.math.log(x)
    return gaussian(lnx, mu, var) - lnx


@check_shapes(
    "x: [broadcast shape...]",
    "p: [broadcast shape...]",
    "return: [shape...]",
)
def bernoulli(x: TensorType, p: TensorType) -> tf.Tensor:
    return tf.math.log(tf.where(tf.equal(x, 1), p, 1 - p))


@check_shapes(
    "x: [broadcast shape...]",
    "lam: [broadcast shape...]",
    "return: [shape...]",
)
def poisson(x: TensorType, lam: TensorType) -> tf.Tensor:
    return x * tf.math.log(lam) - lam - tf.math.lgamma(x + 1.0)


@check_shapes(
    "x: [broadcast shape...]",
    "scale: [broadcast shape...]",
    "return: [shape...]",
)
def exponential(x: TensorType, scale: TensorType) -> tf.Tensor:
    return -x / scale - tf.math.log(scale)


@check_shapes(
    "x: [broadcast shape...]",
    "shape: [broadcast shape...]",
    "scale: [broadcast shape...]",
    "return: [shape...]",
)
def gamma(x: TensorType, shape: TensorType, scale: TensorType) -> tf.Tensor:
    return (
        -shape * tf.math.log(scale)
        - tf.math.lgamma(shape)
        + (shape - 1.0) * tf.math.log(x)
        - x / scale
    )


@check_shapes(
    "x: [broadcast shape...]",
    "mean: [broadcast shape...]",
    "scale: [broadcast shape...]",
    "df: [broadcast shape...]",
    "return: [shape...]",
)
def student_t(x: TensorType, mean: TensorType, scale: TensorType, df: TensorType) -> tf.Tensor:
    df = to_default_float(df)
    const = (
        tf.math.lgamma((df + 1.0) * 0.5)
        - tf.math.lgamma(df * 0.5)
        - 0.5 * (tf.math.log(tf.square(scale)) + tf.math.log(df) + np.log(np.pi))
    )
    return const - 0.5 * (df + 1.0) * tf.math.log(
        1.0 + (1.0 / df) * (tf.square((x - mean) / scale))
    )


@check_shapes(
    "x: [broadcast shape...]",
    "alpha: [broadcast shape...]",
    "beta: [broadcast shape...]",
    "return: [shape...]",
)
def beta(x: TensorType, alpha: TensorType, beta: TensorType) -> tf.Tensor:
    # need to clip x, since log of 0 is nan...
    x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)
    return (
        (alpha - 1.0) * tf.math.log(x)
        + (beta - 1.0) * tf.math.log(1.0 - x)
        + tf.math.lgamma(alpha + beta)
        - tf.math.lgamma(alpha)
        - tf.math.lgamma(beta)
    )


@check_shapes(
    "x: [broadcast shape...]",
    "mu: [broadcast shape...]",
    "sigma: [broadcast shape...]",
    "return: [shape...]",
)
def laplace(x: TensorType, mu: TensorType, sigma: TensorType) -> tf.Tensor:
    return -tf.abs(mu - x) / sigma - tf.math.log(2.0 * sigma)


@check_shapes(
    "x: [D, broadcast N]",
    "mu: [D, broadcast N]",
    "L: [D, D]",
    "return: [N]",
)
def multivariate_normal(x: TensorType, mu: TensorType, L: TensorType) -> tf.Tensor:
    """
    Computes the log-density of a multivariate normal.

    :param x: sample(s) for which we want the density
    :param mu: mean(s) of the normal distribution
    :param L: Cholesky decomposition of the covariance matrix
    :return: log densities
    """

    d = x - mu
    alpha = tf.linalg.triangular_solve(L, d, lower=True)
    num_dims = tf.cast(tf.shape(d)[0], L.dtype)
    p = -0.5 * tf.reduce_sum(tf.square(alpha), 0)
    p -= 0.5 * num_dims * np.log(2 * np.pi)
    p -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

    return p
