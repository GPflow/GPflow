from typing import List

import numpy as np
import tensorflow as tf

from .base import GaussianQuadrature
from ..config import default_float

from ..base import TensorType


def gh_points_and_weights(n_gh: int):
    r"""
    Given the number of Gauss-Hermite points n_gh,
    returns the points z and the weights dz to perform the following
    uni-dimensional gaussian quadrature:

    X ~ N(mean, stddev²)
    E[f(X)] = ∫f(x)p(x)dx = sum_{i=1}^{n_gh} f(mean + stddev*z_i)*dz_i

    :param n_gh: Number of Gauss-Hermite points, integer
    :returns: Points z and weights dz, both tensors with shape [n_gh],
        to compute uni-dimensional gaussian expectation
    """
    z, dz = np.polynomial.hermite.hermgauss(n_gh)
    z = z * np.sqrt(2)
    dz = dz / np.sqrt(np.pi)
    z, dz = z.astype(default_float()), dz.astype(default_float())
    return tf.convert_to_tensor(z), tf.convert_to_tensor(dz)


def list_to_flat_grid(xs: List[TensorType]):
    """
    :param xs: List with d rank-1 Tensors, with shapes N1, N2, ..., Nd
    :return: Tensor with shape [N1*N2*...*Nd, d] representing the flattened
        d-dimensional grid built from the input tensors xs
    """
    return tf.reshape(tf.stack(tf.meshgrid(*xs), axis=-1), (-1, len(xs)))


def reshape_Z_dZ(zs: List[TensorType], dzs: List[TensorType]):
    """
    :param zs: List with d rank-1 Tensors, with shapes N1, N2, ..., Nd
    :param dzs: List with d rank-1 Tensors, with shapes N1, N2, ..., Nd
    :returns: points Z, Tensor with shape [N1*N2*...*Nd, d],
        and weights dZ, Tensor with shape [N1*N2*...*Nd, 1]
    """
    Z = list_to_flat_grid(zs)
    dZ = tf.reduce_prod(list_to_flat_grid(dzs), axis=-1, keepdims=True)
    return Z, dZ


def repeat_as_list(x: TensorType, n: int):
    """
    :param x: Array/Tensor to be repeated
    :param n: Integer with the number of repetitions
    :return: List of n repetitions of Tensor x
    """
    return [x for _ in range(n)]


def ndgh_points_and_weights(dim: int, n_gh: int):
    r"""
    :param n_gh: number of Gauss-Hermite points, integer
    :param dim: dimension of the multivariate normal, integer
    :returns: points Z, Tensor with shape [n_gh**dim, dim],
        and weights dZ, Tensor with shape [n_gh**dim, 1]
    """
    z, dz = gh_points_and_weights(n_gh)
    zs = repeat_as_list(z, dim)
    dzs = repeat_as_list(dz, dim)
    return reshape_Z_dZ(zs, dzs)


class NDiagGHQuadrature(GaussianQuadrature):
    def __init__(self, dim: int, n_gh: int):
        """
        :param n_gh: number of Gauss-Hermite points, integer
        :param dim: dimension of the multivariate normal, integer
        """
        self.dim = dim
        self.n_gh = n_gh
        self.n_gh_total = n_gh ** dim
        Z, dZ = ndgh_points_and_weights(self.dim, self.n_gh)
        self.Z = tf.ensure_shape(Z, (self.n_gh_total, self.dim))
        self.dZ = tf.ensure_shape(dZ, (self.n_gh_total, 1))

    def _build_X_W(self, mean: TensorType, var: TensorType):
        """
        :param mean: Array/Tensor with shape [b1, b2, ..., bX, dim], usually [N, dim],
            representing the mean of a dim-Variate Gaussian distribution
        :param var: Array/Tensor with shape b1, b2, ..., bX, dim], usually [N, dim],
            representing the variance of a dim-Variate Gaussian distribution
        :return: points X, Tensor with shape [b1, b2, ..., bX, n_gh_total, dim],
            usually [N, n_gh_total, dim],
            and weights W, a Tensor with shape [b1, b2, ..., bX, n_gh_total, 1],
            usually [N, n_gh_total, 1]
        """

        # mean, stddev: [b1, b2, ..., bX, dim], usually [N, dim]
        mean = tf.expand_dims(mean, -2)
        stddev = tf.expand_dims(tf.sqrt(var), -2)
        # mean, stddev: [b1, b2, ..., bX, 1, dim], usually [N, 1, dim]

        X = mean + stddev * self.Z
        W = self.dZ
        # X: [b1, b2, ..., bX, n_gh_total, dim], usually [N, n_gh_total, dim]
        # W: [b1, b2, ..., bX, n_gh_total,   1], usually [N, n_gh_total,   1]

        return X, W
