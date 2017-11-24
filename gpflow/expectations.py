import types

import numpy as np
import tensorflow as tf
from multipledispatch import dispatch

from . import kernels, mean_functions, GPflowError, settings
from .features import InducingFeature
from .quadrature import mvnquad


class ProbabilityDistribution:
    pass


class Gaussian(ProbabilityDistribution):
    def __init__(self, mu, cov):
        self.mu = mu  # N x D
        self.cov = cov  # N x D x D


class DiagonalGaussian(ProbabilityDistribution):
    def __init__(self, mu, vars):
        self.mu = mu  # N x D
        self.vars = vars  # N x D


class Uniform(ProbabilityDistribution):
    def __init__(self, a, b):
        self.a = a  # N x D
        self.b = b


def get_eval_func(obj, feature, slice=None):
    if feature is not None:
        # kernel + feature combination
        if not isinstance(feature, InducingFeature) or not isinstance(obj, kernels.Kern):
            raise GPflowError("If `feature` is supplied, `obj` must be a kernel.")
        return lambda x: tf.transpose(feature.Kuf(obj, x))[slice]
    elif isinstance(obj, mean_functions.MeanFunction):
        return lambda x: obj(x)[slice]
    elif isinstance(obj, kernels.Kern):
        return lambda x: obj.Kdiag(x)
    elif obj is None:
        return lambda _: tf.ones(1, settings.tf_float)
    elif isinstance(obj, types.FunctionType):
        return obj
    else:
        raise NotImplementedError()


@dispatch(object, (InducingFeature, type(None)), object, (InducingFeature, type(None)), Gaussian)
def expectation(obj1, feature1, obj2, feature2, p):
    eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                           get_eval_func(obj2, feature2, np.s_[:, None, :])(x))
    return mvnquad(eval_func, p.mu, p.cov, 20)
