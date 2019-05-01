from functools import reduce

import tensorflow as tf

from . import dispatch
from .. import kernels
from ..features import InducingPoints
from ..probability_distributions import DiagonalGaussian
from ..util import NoneType
from .expectations import expectation


@dispatch.expectation.register(DiagonalGaussian, kernels.Product, NoneType,
                               NoneType, NoneType)
def _E(p, kernel, _, __, ___, nghp=None):
    """
    Compute the expectation:
    <\HadamardProd_i diag(Ki_{X[:, active_dims_i], X[:, active_dims_i]})>_p(X)
        - \HadamardProd_i Ki_{.,.} :: Product kernel
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: N
    """
    if not kernel.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions."
        )  # pragma: no cover

    exps = [expectation(p, k, nghp=nghp) for k in kernel.kernels]
    return reduce(tf.multiply, exps)


@dispatch.expectation.register(DiagonalGaussian, kernels.Product,
                               InducingPoints, NoneType, NoneType)
def _E(p, kernel, feature, __, ___, nghp=None):
    """
    Compute the expectation:
    <\HadamardProd_i Ki_{X[:, active_dims_i], Z[:, active_dims_i]}>_p(X)
        - \HadamardProd_i Ki_{.,.} :: Product kernel
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: NxM
    """
    if not kernel.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions."
        )  # pragma: no cover

    exps = [expectation(p, (k, feature), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.multiply, exps)


@dispatch.expectation.register(DiagonalGaussian, kernels.Product,
                               InducingPoints, kernels.Product, InducingPoints)
def _E(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = < prodK_{Z, x_n} prodK_{x_n, Z} >_p(x_n)
                   = < (\HadamardProd_i Ki_{Z[:, active_dims_i], x[n, active_dims_i]})  <-- Mx1
               1xM -->  (\HadamardProd_j Kj_{x[n, active_dims_j], Z[:, active_dims_j]}) >_p(x_n)  (MxM)

        - \HadamardProd_i Ki_{.,.}, \HadamardProd_j Kj_{.,.} :: Product kernels
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: NxMxM
    """
    if feat1 != feat2:
        raise NotImplementedError("Different features are not supported.")
    if kern1 != kern2:
        raise NotImplementedError(
            "Calculating the expectation over two "
            "different Product kernels is not supported.")

    kernel = kern1
    feature = feat1

    if not kernel.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions."
        )  # pragma: no cover

    exps = [
        expectation(p, (k, feature), (k, feature), nghp=nghp)
        for k in kernel.kernels
    ]
    return reduce(tf.multiply, exps)
