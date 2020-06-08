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


def reshape_Z_dZ(zs, dzs, dim: int):
    Z = tf.reshape(tf.stack(tf.meshgrid(*zs), axis=-1), (-1, dim))
    dZ = tf.reduce_prod(tf.reshape(tf.stack(tf.meshgrid(*dzs), axis=-1), (-1, dim)), axis=-1, keepdims=True)
    return Z, dZ


def ndgh_points_and_weights(dim: int, n_gh: int):
    r"""
    :param n_gh: number of Gauss-Hermite points, integer
    :param dim: dimension of the multivariate normal, integer
    :returns: points Z, with shape [n_gh**dim, dim], and weights dZ, with shape [n_gh**dim, 1]
    """
    z, dz = gh_points_and_weights(n_gh)
    zs = tf.unstack(tf.repeat(tf.expand_dims(z, axis=0), dim, axis=0), axis=0)
    dzs = tf.unstack(tf.repeat(tf.expand_dims(dz, axis=0), dim, axis=0), axis=0)
    return reshape_Z_dZ(zs, dzs, dim)


class NDDiagGHQuadrature(GaussianQuadrature):
    def __init__(self, dim: int, n_gh: int):
        Z, dZ = ndgh_points_and_weights(dim, n_gh)
        self.n_gh_total = n_gh ** dim
        self.Z = tf.convert_to_tensor(Z)
        self.dZ = tf.convert_to_tensor(dZ)

    def _build_X_W(self, mean, var):
        new_shape = (self.n_gh_total,) + tuple([1] * (len(mean.shape) - 1)) + (-1,)
        Z = tf.reshape(self.Z, new_shape)
        dZ = tf.reshape(self.dZ, new_shape)
        # Z : [n_gh_total] + [1]*len(batch) + [dims],   typically [n_gh_total, 1, dims]
        # dZ: [n_gh_total] + [1]*len(batch) + [1],      typically [n_gh_total, 1, 1]

        mean = tf.expand_dims(mean, 0)
        stddev = tf.expand_dims(tf.sqrt(var), 0)
        # mean, stddev: [1] + batch + [dims],           typically [1, N, dims]

        X = mean + stddev * Z
        W = dZ
        # X: [n_gh_total] + batch + [dims],             typically [n_gh_total, N, dims]
        # W: [n_gh_total] + [1]*len(batch) + [1],       typically [n_gh_total, 1, 1]

        return X, W
