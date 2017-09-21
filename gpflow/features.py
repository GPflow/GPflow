import numpy as np
import tensorflow as tf
from functools import singledispatch
from abc import ABCMeta, abstractmethod

from .param import Param, Parameterized
from . import conditionals
from ._settings import settings


float_type = settings.dtypes.float_type


class InducingFeature(Parameterized, metaclass=ABCMeta):
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


@singledispatch
def conditional(feat, kern, Xnew, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Note the changed function signature compared to conditionals.conditional() to allow for dispatch.
    """
    raise NotImplementedError("No implementation for {} found".format(type(feat).__name__))

@conditional.register(InducingPoints)
def _(feat, kern, Xnew, f, full_cov=False, q_sqrt=None, whiten=False):
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
