from abc import ABCMeta, abstractmethod
from functools import singledispatch

import numpy as np
import tensorflow as tf

from . import conditionals, transforms, kernels
from ._settings import settings
from .param import Param, Parameterized
import six

float_type = settings.dtypes.float_type


# class InducingFeature(Parameterized, metaclass=ABCMeta):  # Pure python3. Not ready to support yet.
class InducingFeature(six.with_metaclass(ABCMeta, Parameterized)):
    """
    Abstract base class for inducing features.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of features, relevant for example to determine the
        size of the variational distribution.
        """
        pass

    @abstractmethod
    def Kuu(self, kern):
        """
        Calculates the covariance matrix between features for kernel `kern`.
        """
        pass

    @abstractmethod
    def Kuf(self, kern, Xnew):
        """
        Calculates the covariance matrix with function values at new points
        `Xnew` for kernel `kern`.
        """
        pass


class InducingPoints(InducingFeature):
    """
    Real-space inducing points
    """

    def __init__(self, Z):
        """
        :param Z: the initial positions of the inducing points, size M x D
        """
        super().__init__()
        self.Z = Param(Z)
        self._Mfeat = len(Z)

    def __len__(self):
        return self._Mfeat

    def Kuu(self, kern):
        Kzz = kern.K(self.Z)
        Kzz += settings.numerics.jitter_level * tf.eye(self._Mfeat, dtype=float_type)
        return Kzz

    def Kuf(self, kern, Xnew):
        Kzx = kern.K(self.Z, Xnew)
        return Kzx


class Multiscale(InducingFeature):
    r"""
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
        super().__init__()
        self.Z = Param(Z)  # Multi-scale feature centres
        self.scales = Param(scales, transform=transforms.positive)  # Multi-scale feature widths (std. dev. of Gaussian)
        self._M = len(Z)

    def __len__(self):
        return self._M

    def _cust_square_dist(self, A, B, sc):
        return tf.reduce_sum(tf.square((tf.expand_dims(A, 1) - tf.expand_dims(B, 0)) / sc), 2)

    def Kuf(self, kern, Xnew):
        if isinstance(kern, kernels.RBF):
            Xnew, _ = kern._slice(Xnew, None)
            Zmu, Zlen = kern._slice(self.Z, self.scales)
            idlengthscales = kern.lengthscales + Zlen
            d = self._cust_square_dist(Xnew, Zmu, idlengthscales)
            return tf.transpose(self.variance * tf.exp(-d / 2) *
                                tf.reshape(tf.reduce_prod(kern.lengthscales / idlengthscales, 1), (1, -1)))
        else:
            raise NotImplementedError("Multiscale features not implemented for `%s`." % str(type(kern)))

    def Kuu(self, kern):
        if isinstance(kern, kernels.RBF):
            Zmu, Zlen = kern._slice(self.Z, self.scales)
            idlengthscales2 = tf.square(kern.lengthscales + Zlen)
            sc = tf.sqrt(
                tf.expand_dims(idlengthscales2, 0) + tf.expand_dims(idlengthscales2, 1) - tf.square(kern.lengthscales))
            d = self._cust_square_dist(Zmu, Zmu, sc)
            return self.variance * tf.exp(-d / 2) * tf.reduce_prod(kern.lengthscales / sc, 2)
        else:
            raise NotImplementedError("Multiscale features not implemented for `%s`." % str(type(kern)))


@singledispatch
def conditional(feat, kern, Xnew, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Note the changed function signature compared to conditionals.conditional()
    to allow for single dispatch on the first argument.
    """
    raise NotImplementedError("No implementation for {} found".format(type(feat).__name__))


@conditional.register(InducingPoints)
@conditional.register(Multiscale)
def default_feature_conditional(feat, kern, Xnew, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Uses the same code path as conditionals.conditional(), except Kuu/Kuf
    matrices are constructed using the feature.
    To use this with features defined in external modules, register your
    feature class using
    >>> gpflow.features.conditional.register(YourFeatureClass,
    ...             gpflow.features.default_feature_conditional)
    """
    return conditionals.feature_conditional(Xnew, feat, kern, f, full_cov=full_cov, q_sqrt=q_sqrt, whiten=whiten)


def inducingpoint_wrapper(feat, Z):
    """
    Backwards-compatibility wrapper for real-space inducing points.
    """
    if feat is not None and Z is not None:
        raise ValueError("Cannot pass both an InducingFeature instance and Z values")
    elif feat is None and Z is None:
        raise ValueError("You must pass either an InducingFeature instance or Z values")
    elif Z is not None:
        feat = InducingPoints(Z)
    elif isinstance(feat, np.ndarray):
        feat = InducingPoints(feat)
    else:
        assert isinstance(feat, InducingFeature)
    return feat
