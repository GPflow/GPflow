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
# limitations under the License.

from abc import abstractmethod
import warnings

import numpy as np
import tensorflow as tf

from . import transforms, kernels, settings
from .decors import params_as_tensors, params_as_tensors_for
from .params import Parameter, Parameterized
from .dispatch import dispatch


logger = settings.logger()


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

    def Kuu(self, kern, jitter=0.0):
        """
        Calculates the covariance matrix between features for kernel `kern`.

        Return shape M x M
        M = len(feat)
        """
        warnings.warn('Please replace feature.Kuu(kernel) with Kuu(feature, kernel)',
                      DeprecationWarning)
        return Kuu(self, kern, jitter=jitter)

    def Kuf(self, kern, Xnew):
        """
        Calculates the covariance matrix with function values at new points
        `Xnew` for kernel `kern`.

        Return shape M x N
        M = len(feat)
        N = len(Xnew)
        """
        warnings.warn('Please replace feature.Kuf(kernel, Xnew) with Kuf(feature, kernel, Xnew)',
                      DeprecationWarning)
        return Kuf(self, kern, Xnew)


class InducingPointsBase(InducingFeature):
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


class InducingPoints(InducingPointsBase):
    pass

@dispatch(InducingPoints, kernels.Kernel)
def Kuu(feat, kern, *, jitter=0.0):
    with params_as_tensors_for(feat):
        Kzz = kern.K(feat.Z)
        Kzz += jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)
    return Kzz

@dispatch(InducingPoints, kernels.Kernel, object)
def Kuf(feat, kern, Xnew):
    with params_as_tensors_for(feat):
        Kzx = kern.K(feat.Z, Xnew)
    return Kzx


class Multiscale(InducingPointsBase):
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

    @staticmethod
    def _cust_square_dist(A, B, sc):
        """
        Custom version of _square_dist that allows sc to provide per-datapoint length
        scales. sc: N x M x D.
        """
        return tf.reduce_sum(tf.square((tf.expand_dims(A, 1) - tf.expand_dims(B, 0)) / sc), 2)


@dispatch(Multiscale, kernels.RBF, object)
def Kuf(feat, kern, Xnew):
    with params_as_tensors_for(feat, kern):
        Xnew, _ = kern._slice(Xnew, None)
        Zmu, Zlen = kern._slice(feat.Z, feat.scales)
        idlengthscales = kern.lengthscales + Zlen
        d = feat._cust_square_dist(Xnew, Zmu, idlengthscales)
        Kuf = tf.transpose(kern.variance * tf.exp(-d / 2) *
                           tf.reshape(tf.reduce_prod(kern.lengthscales / idlengthscales, 1),
                                      (1, -1)))
    return Kuf

@dispatch(Multiscale, kernels.RBF)
def Kuu(feat, kern, *, jitter=0.0):
    with params_as_tensors_for(feat, kern):
        Zmu, Zlen = kern._slice(feat.Z, feat.scales)
        idlengthscales2 = tf.square(kern.lengthscales + Zlen)
        sc = tf.sqrt(
            tf.expand_dims(idlengthscales2, 0) + tf.expand_dims(idlengthscales2, 1) - tf.square(
                kern.lengthscales))
        d = feat._cust_square_dist(Zmu, Zmu, sc)
        Kzz = kern.variance * tf.exp(-d / 2) * tf.reduce_prod(kern.lengthscales / sc, 2)
        Kzz += jitter * tf.eye(len(feat), dtype=settings.float_type)
    return Kzz


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
