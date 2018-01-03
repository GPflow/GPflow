from __future__ import print_function, absolute_import

import itertools

import numpy as np
import tensorflow as tf

from . import settings
from .core.errors import GPflowError


def hermgauss(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = x.astype(settings.float_type), w.astype(settings.float_type)
    return x, w


def mvhermgauss(H, D):
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


def mvnquad(func, means, covs, H, Din=None, Dout=None):
    """
    Computes N Gaussian expectation integrals of a single function 'f'
    using Gauss-Hermite quadrature.
    :param f: integrand function. Takes one input of shape ?xD.
    :param means: NxD
    :param covs: NxDxD
    :param H: Number of Gauss-Hermite evaluation points.
    :param Din: Number of input dimensions. Needs to be known at call-time.
    :param Dout: Number of output dimensions. Defaults to (). Dout is assumed
    to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout).
    :return: quadratures (N,*Dout)
    """
    # Figure out input shape information
    if Din is None:
        Din = means.shape[1] if type(means.shape) is tuple else means.shape[1].value

    if Din is None:
        raise GPflowError("If `Din` is passed as `None`, `means` must have a known shape. "
                          "Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
                          "is problematic. Consider using your own session.")  # pragma: no cover

    xn, wn = mvhermgauss(H, Din)
    N = tf.shape(means)[0]

    # transform points based on Gaussian parameters
    cholXcov = tf.cholesky(covs)  # NxDxD
    Xt = tf.matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), transpose_b=True)  # NxDxH**D
    X = 2.0 ** 0.5 * Xt + tf.expand_dims(means, 2)  # NxDxH**D
    Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, Din))  # (H**D*N)xD

    # perform quadrature
    fevals = func(Xr)
    if Dout is None:
        Dout = tuple((d if type(d) is int else d.value) for d in fevals.shape[1:])

    if any([d is None for d in Dout]):
        raise GPflowError("If `Dout` is passed as `None`, the output of `func` must have known "
                          "shape. Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
                          "is problematic. Consider using your own session.")  # pragma: no cover
    fX = tf.reshape(fevals, (H ** Din, N,) + Dout)
    wr = np.reshape(wn * np.pi ** (-Din * 0.5),
                    (-1,) + (1,) * (1 + len(Dout)))
    return tf.reduce_sum(fX * wr, 0)
