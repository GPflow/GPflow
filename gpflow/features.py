# Copyright 2017 st--, Mark van der Wilk
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

from abc import abstractmethod
from functools import singledispatch

import numpy as np
import tensorflow as tf

from . import conditionals, transforms, kernels, decors, settings
from .params import Parameter, Parameterized


class InducingFeature(Parameterized):
    """
    Abstract base class for inducing features.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of features, relevant for example to determine the
        size of the variational distribution.
        """
        raise NotImplementedError()

    @abstractmethod
    def Kuu(self, kern, jitter=0.0):
        """
        Calculates the covariance matrix between features for kernel `kern`.
        """
        raise NotImplementedError()

    @abstractmethod
    def Kuf(self, kern, Xnew):
        """
        Calculates the covariance matrix with function values at new points
        `Xnew` for kernel `kern`.
        """
        raise NotImplementedError()


class InducingPoints(InducingFeature):
    """
    Real-space inducing points
    """

    def __init__(self, Z):
        """
        :param Z: the initial positions of the inducing points, size M x D
        """
        super().__init__()
        self.Z = Parameter(Z, dtype=settings.float_type)

    def __len__(self):
        return self.Z.shape[0]

    @decors.params_as_tensors
    def Kuu(self, kern, jitter=0.0):
        Kzz = kern.K(self.Z)
        Kzz += jitter * tf.eye(len(self), dtype=settings.dtypes.float_type)
        return Kzz

    @decors.params_as_tensors
    def Kuf(self, kern, Xnew):
        Kzx = kern.K(self.Z, Xnew)
        return Kzx


class Multiscale(InducingPoints):
    """
    Multi-scale inducing features
    Originally proposed in

    ::

      @incollection{NIPS2009_3876,
        title = {Inter-domain Gaussian Processes for Sparse Inference using Inducing Features},
        author = {Miguel L\'{a}zaro-Gredilla and An\'{\i}bal Figueiras-Vidal},
        booktitle = {Advances in Neural Information Processing Systems 22},
        year = {2009},
      }

    """

    def __init__(self, Z, scales):
        super().__init__(Z)
        self.scales = Parameter(scales,
                                transform=transforms.positive)  # Multi-scale feature widths (std. dev. of Gaussian)
        if self.Z.shape != scales.shape:
            raise ValueError("Input locations `Z` and `scales` must have the same shape.")  # pragma: no cover

    def _cust_square_dist(self, A, B, sc):
        """
        Custom version of _square_dist that allows sc to provide per-datapoint length
        scales. sc: N x M x D.
        """
        return tf.reduce_sum(tf.square((tf.expand_dims(A, 1) - tf.expand_dims(B, 0)) / sc), 2)

    @decors.params_as_tensors
    def Kuf(self, kern, Xnew):
        if isinstance(kern, kernels.RBF):
            with decors.params_as_tensors_for(kern):
                Xnew, _ = kern._slice(Xnew, None)
                Zmu, Zlen = kern._slice(self.Z, self.scales)
                idlengthscales = kern.lengthscales + Zlen
                d = self._cust_square_dist(Xnew, Zmu, idlengthscales)
                Kuf = tf.transpose(kern.variance * tf.exp(-d / 2) *
                                   tf.reshape(tf.reduce_prod(kern.lengthscales / idlengthscales, 1),
                                              (1, -1)))
            return Kuf
        else:
            raise NotImplementedError(
                "Multiscale features not implemented for `%s`." % str(type(kern)))

    @decors.params_as_tensors
    def Kuu(self, kern, jitter=0.0):
        if isinstance(kern, kernels.RBF):
            with decors.params_as_tensors_for(kern):
                Zmu, Zlen = kern._slice(self.Z, self.scales)
                idlengthscales2 = tf.square(kern.lengthscales + Zlen)
                sc = tf.sqrt(
                    tf.expand_dims(idlengthscales2, 0) + tf.expand_dims(idlengthscales2, 1) - tf.square(
                        kern.lengthscales))
                d = self._cust_square_dist(Zmu, Zmu, sc)
                Kzz = kern.variance * tf.exp(-d / 2) * tf.reduce_prod(kern.lengthscales / sc, 2)
                Kzz += jitter * tf.eye(len(self), dtype=settings.float_type)
            return Kzz
        else:
            raise NotImplementedError(
                "Multiscale features not implemented for `%s`." % str(type(kern)))


@singledispatch
def conditional(feat, kern, Xnew, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Note the changed function signature compared to conditionals.conditional()
    to allow for single dispatch on the first argument.
    """
    raise NotImplementedError("No implementation for {} found".format(type(feat).__name__))


@conditional.register(InducingPoints)
@conditional.register(Multiscale)
def default_feature_conditional(feat, kern, Xnew, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Uses the same code path as conditionals.conditional(), except Kuu/Kuf
    matrices are constructed using the feature.
    To use this with features defined in external modules, register your
    feature class using
    >>> gpflow.features.conditional.register(YourFeatureClass,
    ...             gpflow.features.default_feature_conditional)
    """
    return conditionals.feature_conditional(Xnew, feat, kern, f, full_cov=full_cov, q_sqrt=q_sqrt,
                                            white=white)


def inducingpoint_wrapper(feat, Z):
    """
    Models which used to take only Z can now pass `feat` and `Z` to this method. This method will
    check for consistency and return the correct feature. This allows backwards compatibility in
    for the methods.
    """
    if feat is not None and Z is not None:
        raise ValueError("Cannot pass both an InducingFeature instance and Z values")  # pragma: no cover
    elif feat is None and Z is None:
        raise ValueError("You must pass either an InducingFeature instance or Z values")  # pragma: no cover
    elif Z is not None:
        feat = InducingPoints(Z)
    elif isinstance(feat, np.ndarray):
        feat = InducingPoints(feat)
    else:
        assert isinstance(feat, InducingFeature)  # pragma: no cover
    return feat
