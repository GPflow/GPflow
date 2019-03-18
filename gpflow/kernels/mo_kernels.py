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


class Mok(Kernel):

    @abc.abstractmethod
    def K(self, X, Y=None, full_output_cov=True):
        pass

    @abc.abstractmethod
    def K_diag(self, X, full_output_cov=True):
        pass

    def __call__(self, X, Y=None, full=True, full_output_cov=True):
        if not full and Y is not None:
            raise ValueError("Ambiguous inputs: `diagonal` and `y` are not compatible.")
        if not full:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, Y, full_output_cov=full_output_cov)
        pass


class SharedIndependentMok(Mok):
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
            Ks = tf.tile(K[..., None], [1, 1, self.P])  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Ks), [0, 2, 1, 3])  # [N, P, N]2 x P
        else:
            return tf.tile(K[None, ...], [self.P, 1, 1])  # [P, N, N]2

    def K_diag(self, X, full_output_cov=True):
        K = self.kern(X)  # N
        Ks = tf.tile(K[:, None], [1, self.P])  # N x P
        return tf.linalg.diag(Ks) if full_output_cov else Ks  # [N, P, P] or N x P


class SeparateIndependentMok(Combination, Mok):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels):
        super().__init__(kernels)

    def K(self, X, X2=None, full_output_cov=True):
        if full_output_cov:
            Kxxs = tf.stack([k(X, X2) for k in self.kernels], axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N]2 x P
        else:
            return tf.stack([k(X, X2) for k in self.kernels], axis=0)  # [P, N, N]2

    def K_diag(self, X, full_output_cov=False):
        stacked = tf.stack([k(X) for k in self.kernels], axis=1)  # N x P
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  N x P


class SeparateMixedMok(Combination, Mok):
    """
    Linear mixing of the latent GPs to form the output
    """

    def __init__(self, kernels, W):
        super().__init__(kernels)
        self.W = Parameter(W)  # P x L

    def Kgg(self, X, X2):
        return tf.stack([k(X, X2) for k in self.kernels], axis=0)  # [L, N, N]2

    def K(self, X, X2=None, full_output_cov=True):
        Kxx = self.Kgg(X, X2)  # [L, N, N]2
        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None]  # [P, L, N, N]2
        if full_output_cov:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # [P, P, N, N]2
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, N]2 x P
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
            return tf.reduce_sum(self.W[:, :, None, None] * KxxW, [1])  # [P, N, N]2

    def K_diag(self, X, full_output_cov=True):
        K = tf.stack([k(X) for k in self.kernels], axis=1)  # N x L
        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
            Wt = tf.transpose(self.W)  # L x P
            return tf.reduce_sum(K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :],
                                 axis=1)  # [N, P, P]
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # N x P
            return tf.matmul(K, self.W ** 2.0, transpose_b=True)  # N x L  *  L x P  ->  N x P
