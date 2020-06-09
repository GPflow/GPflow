import numpy as np
import tensorflow as tf

from .base import GaussianQuadrature
from ..config import default_float


def gh_points_and_weights(n_gh: int):
    r"""
    Given the number of Gauss-Hermite points n_gh,
    returns the points z and the weights dz to perform the following
    uni-dimensional gaussian quadrature:

    q(x) = N(mu, sigma²)

    E_{X~q(x)}[f(X)] = ∫ f(x)q(x)dx = \sum f(mu + sigma*z_k)*dz_k

    :param n_gh: number of Gauss-Hermite points, integer
    :returns: points z and weights dz to compute uni-dimensional gaussian expectation, tuple
    """
    z, dz = np.polynomial.hermite.hermgauss(n_gh)
    z = z * np.sqrt(2)
    dz = dz / np.sqrt(np.pi)
    z, dz = z.astype(default_float()), dz.astype(default_float())
    return tf.convert_to_tensor(z), tf.convert_to_tensor(dz)


def list_to_flat_grid(xs):
    return tf.reshape(tf.stack(tf.meshgrid(*xs), axis=-1), (-1, len(xs)))


def reshape_Z_dZ(zs, dzs):
    Z = list_to_flat_grid(zs)
    dZ = tf.reduce_prod(list_to_flat_grid(dzs), axis=-1, keepdims=True)
    return Z, dZ


def repeat_as_list(x, n):
    return tf.unstack(tf.repeat(tf.expand_dims(x, axis=0), n, axis=0), axis=0)


def ndgh_points_and_weights(dim: int, n_gh: int):
    r"""
    :param n_gh: number of Gauss-Hermite points, integer
    :param dim: dimension of the multivariate normal, integer
    :returns: points Z, with shape [n_gh**dim, dim], and weights dZ, with shape [n_gh**dim, 1]
    """
    z, dz = gh_points_and_weights(n_gh)
    zs = repeat_as_list(z, dim)
    dzs = repeat_as_list(dz, dim)
    return reshape_Z_dZ(zs, dzs)


class NDiagGHQuadrature(GaussianQuadrature):
    def __init__(self, dim: int, n_gh: int):
        Z, dZ = ndgh_points_and_weights(dim, n_gh)
        self.n_gh_total = n_gh ** dim
        self.Z = tf.convert_to_tensor(Z)
        self.dZ = tf.convert_to_tensor(dZ)
        #  Z: [n_gh_total, dims]
        # dZ: [n_gh_total,    1]

    def _build_X_W(self, mean, var):
        # mean, stddev: batch + [dims], typically [N, dims]
        mean = tf.expand_dims(mean, -2)
        stddev = tf.expand_dims(tf.sqrt(var), -2)
        # mean, stddev: batch + [1, dims], typically [N, 1, dims]

        X = mean + stddev * self.Z
        W = self.dZ
        # X: batch + [n_gh_total, dims], typically [N, n_gh_total, dims]
        # W: batch + [n_gh_total,    1], typically [N, n_gh_total,    1]

        return X, W
