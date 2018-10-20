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


import abc

import tensorflow as tf

from ..base import Parameter
from .base import Combination, Kernel


class Mok(metaclass=abc.ABCMeta):
    pass


class SharedIndependentMok(Kernel, Mok):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.

    Note: this class is created only for testing and comparison purposes.
    Use `gpflow.kernels` instead for more efficient code.
    """
    def __init__(self, kern: Kernel, output_dimensionality, name=None):
        super().__init__(name)
        self.kern = kern
        self.P = output_dimensionality

    def K(self, X, X2=None, full_output_cov=True):
        K = self.kern(X, X2)  # N x N2
        if full_output_cov:
            Ks = tf.tile(K[..., None], [1, 1, self.P])  # N x N2 x P
            return tf.transpose(tf.matrix_diag(Ks), [0, 2, 1, 3])  # N x P x N2 x P
        else:
            return tf.tile(K[None, ...], [self.P, 1, 1])  # P x N x N2

    def Kdiag(self, X, full_output_cov=True):
        K = self.kern(X)  # N
        Ks = tf.tile(K[:, None], [1, self.P])  # N x P
        return tf.matrix_diag(Ks) if full_output_cov else Ks  # N x P x P or N x P


class SeparateIndependentMok(Combination, Mok):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """
    def __init__(self, kernels, name=None):
        super().__init__(kernels, name)

    def K(self, X, X2=None, full_output_cov=True):
        if full_output_cov:
            Kxxs = tf.stack([k(X, X2) for k in self.kernels], axis=2)  # N x N2 x P
            return tf.transpose(tf.matrix_diag(Kxxs), [0, 2, 1, 3])  # N x P x N2 x P
        else:
            return tf.stack([k(X, X2) for k in self.kernels], axis=0)  # P x N x N2

    def Kdiag(self, X, full_output_cov=False):
        stacked = tf.stack([k(X) for k in self.kernels], axis=1)  # N x P
        return tf.matrix_diag(stacked) if full_output_cov else stacked  # N x P x P  or  N x P


class SeparateMixedMok(Combination, Mok):
    """
    Linear mixing of the latent GPs to form the output
    """

    def __init__(self, kernels, W, name=None):
        super().__init__(kernels, name)
        self.W = Parameter(W)  # P x L

    def Kgg(self, X, X2):
        return tf.stack([k(X, X2) for k in self.kernels], axis=0)  # L x N x N2

    def K(self, X, X2=None, full_output_cov=True):
        Kxx = self.Kgg(X, X2)  # L x N x N2
        KxxW = Kxx[None, :, :, :] * self.W()[:, :, None, None]  # P x L x N x N2
        if full_output_cov:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
            WKxxW = tf.tensordot(self.W(), KxxW, [[1], [1]])  # P x P x N x N2
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # N x P x N2 x P
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
            return tf.reduce_sum(self.W()[:, :, None, None] * KxxW, [1])  # P x N x N2

    def Kdiag(self, X, full_output_cov=True):
        K = tf.stack([k(X) for k in self.kernels], axis=1)  # N x L
        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # N x P x P
            Wt = tf.transpose(self.W())  # L x P
            return tf.reduce_sum(K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :], axis=1)  # N x P x P
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # N x P
            return tf.matmul(K, self.W() ** 2.0, transpose_b=True)  # N x L  *  L x P  ->  N x P
