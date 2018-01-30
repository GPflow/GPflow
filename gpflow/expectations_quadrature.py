# Copyright 2018 the GPflow authors.
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
# limitations under the License.from __future__ import print_function

import types
import warnings

import numpy as np
import functools
import tensorflow as tf
from multipledispatch import dispatch
from functools import partial

from . import kernels, mean_functions
from .probability_distributions import Gaussian, DiagonalGaussian, MarkovGaussian
from .features import InducingFeature
from .quadrature import mvnquad

# By default multipledispatch uses a global namespace in multipledispatch.core.global_namespace.
# We define our own GPflow namespace to avoid any conflict which may arise.
gpflow_md_namespace = dict()
dispatch = partial(dispatch, namespace=gpflow_md_namespace)


def quadrature_fallback(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError as e:
            print(str(e))
            return _quadrature_expectation(*args)

    return wrapper


def quadrature_expectation(p, obj1, obj2=None):
    if isinstance(obj1, tuple):
        obj1, feat1 = obj1
    else:
        feat1 = None

    if isinstance(obj2, tuple):
        obj2, feat2 = obj2
    else:
        feat2 = None

    return _quadrature_expectation(p, obj1, feat1, obj2, feat2)


# Generic case, quadrature method

def get_eval_func(obj, feature, slice=np.s_[...]):
    if feature is not None:
        # kernel + feature combination
        if not isinstance(feature, InducingFeature) or not isinstance(obj, kernels.Kernel):
            raise TypeError("If `feature` is supplied, `obj` must be a kernel.")
        return lambda x: tf.transpose(feature.Kuf(obj, x))[slice]
    elif isinstance(obj, mean_functions.MeanFunction):
        return lambda x: obj(x)[slice]
    elif isinstance(obj, kernels.Kernel):
        return lambda x: obj.Kdiag(x)
    else:
        raise NotImplementedError()


@dispatch((Gaussian, DiagonalGaussian), object, (InducingFeature, type(None)), object, (InducingFeature, type(None)))
def _quadrature_expectation(p, obj1, feature1, obj2, feature2, H=100):
    warnings.warn("Quadrature is used to calculate the expectation. This means that "
                  "an analytical implementations is not available for the given combination.")
    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature2)(x)
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(x))

    cov = tf.matrix_diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov
    return mvnquad(eval_func, p.mu, cov, H)


@dispatch(MarkovGaussian, object, (InducingFeature, type(None)), object, (InducingFeature, type(None)))
def _quadrature_expectation(p, obj1, feature1, obj2, feature2, H=100):
    warnings.warn("Quadrature is used to calculate the expectation. This means that "
                  "an analytical implementations is not available for the given combination.")
    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(tf.split(x, 2, 1)[0])
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature2)(tf.split(x, 2, 1)[1])
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(tf.split(x, 2, 1)[0]) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(tf.split(x, 2, 1)[1]))

    mu = tf.concat((p.mu[:-1, :], p.mu[1:, :]), 1)  # Nx2D
    cov_top = tf.concat((p.cov[0, :-1, :, :], p.cov[1, :-1, :, :]), 2)  # NxDx2D
    cov_bottom = tf.concat((tf.matrix_transpose(p.cov[1, :-1, :, :]), p.cov[0, 1:, :, :]), 2)
    cov = tf.concat((cov_top, cov_bottom), 1)  # Nx2Dx2D

    return mvnquad(eval_func, mu, cov, H)
