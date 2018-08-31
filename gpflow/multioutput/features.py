# Copyright 2018 GPflow authors
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

import tensorflow as tf

from .. import settings
from ..dispatch import dispatch
from ..features import InducingPoints, InducingFeature, Kuu, Kuf
from ..decors import params_as_tensors_for
from ..params import ParamList
from .kernels import Mok, SharedIndependentMok, SeparateIndependentMok, SeparateMixedMok


logger = settings.logger()


class Mof(InducingFeature):
    """
    Class used to indicate that we are dealing with
    features that are used for multiple outputs.
    """
    pass


class SharedIndependentMof(Mof):
    """
    Same feature is used for each output.
    """
    def __init__(self, feat):
        Mof.__init__(self)
        self.feat = feat

    def __len__(self):
        return len(self.feat)


class SeparateIndependentMof(Mof):
    """
    A different feature is used for each output.
    Note: each feature should have the same number of points, M.
    """
    def __init__(self, feat_list):
        Mof.__init__(self)
        self.feat_list = ParamList(feat_list)

    def __len__(self):
        return len(self.feat_list[0])


class MixedKernelSharedMof(SharedIndependentMof):
    """
    This Mof is used in combination with the `SeparateMixedMok`.
    Using this feature with the `SeparateMixedMok` leads to the most efficient code.
    """
    pass

class MixedKernelSeparateMof(SeparateIndependentMof):
    """
    This Mof is used in combination with the `SeparateMixedMok`.
    Using this feature with the `SeparateMixedMok` leads to the most efficient code.
    """
    pass


# ---
# Kuf
# ---

def debug_kuf(feat, kern):
    msg = "Dispatch to Kuf(feat: {}, kern: {})"
    logger.debug(msg.format(
        feat.__class__.__name__,
        kern.__class__.__name__))

@dispatch(InducingPoints, Mok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    return kern.K(feat.Z, Xnew, full_output_cov=True)  #  M x P x N x P


@dispatch(SharedIndependentMof, SharedIndependentMok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    return Kuf(feat.feat, kern.kern, Xnew)  # M x N


@dispatch(SeparateIndependentMof, SharedIndependentMok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(f, kern.kern, Xnew) for f in feat.feat_list], axis=0)  # L x M x N


@dispatch(SharedIndependentMof, SeparateIndependentMok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(feat.feat, k, Xnew) for k in kern.kernels], axis=0)  # L x M x N


@dispatch(SeparateIndependentMof, SeparateIndependentMok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(f, k, Xnew) for f, k in zip(feat.feat_list, kern.kernels)], axis=0)  # L x M x N


@dispatch((SeparateIndependentMof, SharedIndependentMof), SeparateMixedMok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    kuf_impl = Kuf.dispatch(type(feat), SeparateIndependentMok, object)
    K = tf.transpose(kuf_impl(feat, kern, Xnew), [1, 0, 2])  # M x L x N
    with params_as_tensors_for(kern):
        return K[:, :, :, None] * tf.transpose(kern.W)[None, :, None, :]  # M x L x N x P


@dispatch(MixedKernelSharedMof, SeparateMixedMok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(feat.feat, k, Xnew) for k in kern.kernels], axis=0)  # L x M x N

@dispatch(MixedKernelSeparateMof, SeparateMixedMok, object)
def Kuf(feat, kern, Xnew):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(f, k, Xnew) for f, k in zip(feat.feat_list, kern.kernels)], axis=0)  # L x M x N

# ---
# Kuu
# ---


def debug_kuu(feat, kern, jitter):
    msg = "Dispatch to Kuu(feat: {}, kern: {}) with jitter={}"
    logger.debug(msg.format(
        feat.__class__.__name__,
        kern.__class__.__name__,
        jitter))


@dispatch(InducingPoints, Mok)
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern.K(feat.Z, full_output_cov=True)  # M x P x M x P
    M = tf.shape(Kmm)[0] * tf.shape(Kmm)[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=settings.float_type), tf.shape(Kmm))
    return Kmm + jittermat


@dispatch(SharedIndependentMof, SharedIndependentMok)
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = Kuu(feat.feat, kern.kern)  # M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type) * jitter
    return Kmm + jittermat


@dispatch(SharedIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(feat.feat, k) for k in kern.kernels], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
    return Kmm + jittermat


@dispatch(SeparateIndependentMof, SharedIndependentMok)
def Kuu(feat, kern, *, jitter):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(f, kern.kern) for f in feat.feat_list], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
    return Kmm + jittermat


@dispatch((SeparateIndependentMof,MixedKernelSeparateMof), (SeparateIndependentMok, SeparateMixedMok))
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(f, k) for f, k in zip(feat.feat_list, kern.kernels)], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
    return Kmm + jittermat


@dispatch(MixedKernelSharedMof, SeparateMixedMok)
def Kuu(feat, kern, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(feat.feat, k) for k in kern.kernels], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
    return Kmm + jittermat
