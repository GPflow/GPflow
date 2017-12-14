# Copyright 2017 the GPflow authors.
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

import numpy as np
import functools
import tensorflow as tf
from multipledispatch import dispatch

from . import kernels, mean_functions
from .probability_distributions import Gaussian, DiagonalGaussian
from .features import InducingFeature
from .quadrature import mvnquad


def quadrature_fallback(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError as e:
            print(str(e))
            return expectation_quad(*args)

    return wrapper


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


@dispatch(Gaussian, object, (InducingFeature, type(None)), object, (InducingFeature, type(None)))
def _expectation(p, obj1, feature1, obj2, feature2, H=40):
    print("Quad")
    print("H", H)
    # warnings.warn("Quadrature is being used to calculate expectation")
    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature2)(x)
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(x))

    return mvnquad(eval_func, p.mu, p.cov, H)


expectation_quad = _expectation.dispatch(Gaussian, object, type(None), object, type(None))
