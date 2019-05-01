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
    """
    Multi Output Kernel class.
    This kernel can represent correlation between outputs of different datapoints.
    Therefore, subclasses of Mok should implement `K` which returns:
    - [N, P, N, P] if full_output_cov = True
    - [P, N, N] if full_output_cov = False
    and `K_diag` returns:
    - [N, P, P] if full_output_cov = True
    - [N, P] if full_output_cov = False
    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """

    @abc.abstractmethod
    def K(self, X, Y=None, full_output_cov=True, presliced=False):
        """
        Returns the correlation of f(X1) and f(Y), where f(.) can be multi-dimensional.
        :param X: data matrix, [1, D]
        :param Y: data matrix, [2, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X1), f(Y)] with shape
        - [1, P, 2, P] if `full_output_cov` = True
        - [P, 1, 2] if `full_output_cov` = False
        """
        pass

    @abc.abstractmethod
    def K_diag(self, X, full_output_cov=True, presliced=False):
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.
        :param X: data matrix, [N, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)] with shape
        - [N, P, N, P] if `full_output_cov` = True
        - [N, P] if `full_output_cov` = False
        """
        pass

    def __call__(self,
                 X,
                 Y=None,
                 full=False,
                 full_output_cov=True,
                 presliced=False):
        if not full and Y is not None:
            raise ValueError(
                "Ambiguous inputs: `diagonal` and `y` are not compatible.")
        if not full:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, Y, full_output_cov=full_output_cov)


class SharedIndependentMok(Mok):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.
    Note: this class is created only for testing and comparison purposes.
    Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(self, kernel: Kernel, output_dimensionality: int):
        super().__init__()
        self.kernel = kernel
        self.P = output_dimensionality

    def K(self, X, Y=None, full_output_cov=True, presliced=False):
        K = self.kernel.K(X, Y)  # [N, 2]
        if full_output_cov:
            Ks = tf.tile(K[..., None], [1, 1, self.P])  # [N, 2, P]
            return tf.transpose(tf.linalg.diag(Ks),
                                [0, 2, 1, 3])  # [N, P, 2, P]
        else:
            return tf.tile(K[None, ...], [self.P, 1, 1])  # [P, N, 2]

    def K_diag(self, X, full_output_cov=True, presliced=False):
        K = self.kernel.K_diag(X)  # N
        Ks = tf.tile(K[:, None], [1, self.P])  # [N, P]
        return tf.linalg.diag(
            Ks) if full_output_cov else Ks  # [N, P, P] or [N, P]


class SeparateIndependentMok(Mok, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, name=None):
        Combination.__init__(self, kernels, name)

    def K(self, X, Y=None, full_output_cov=True, presliced=False):
        if full_output_cov:
            Kxxs = tf.stack([k.K(X, Y) for k in self.kernels],
                            axis=2)  # [N, 2, P]
            return tf.transpose(tf.linalg.diag(Kxxs),
                                [0, 2, 1, 3])  # [N, P, 2, P]
        else:
            return tf.stack([k.K(X, Y) for k in self.kernels],
                            axis=0)  # [P, N, 2]

    def K_diag(self, X, full_output_cov=False, presliced=False):
        stacked = tf.stack([k.K_diag(X) for k in self.kernels],
                           axis=1)  # [N, P]
        return tf.linalg.diag(
            stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]


class SeparateMixedMok(Mok, Combination):
    """
    Linear mixing of the latent GPs to form the output.
    """

    def __init__(self, kernels, W, name=None):
        Combination.__init__(self, kernels, name)
        self.W = Parameter(W)  # [P, L]

    def Kgg(self, X, Y):
        return tf.stack([k.K(X, Y) for k in self.kernels], axis=0)  # [L, N, 2]

    def K(self, X, Y=None, full_output_cov=True, presliced=False):
        Kxx = self.Kgg(X, Y)  # [L, N, 2]
        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None]  # [P, L, N, 2]
        if full_output_cov:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # [P, P, N, 2]
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, 2, P]
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
            return tf.reduce_sum(self.W[:, :, None, None] * KxxW,
                                 [1])  # [P, N, 2]

    def K_diag(self, X, full_output_cov=True, presliced=False):
        K = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, L]
        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
            Wt = tf.transpose(self.W)  # [L, P]
            return tf.reduce_sum(K[:, :, None, None] * Wt[None, :, :, None] *
                                 Wt[None, :, None, :],
                                 axis=1)  # [N, P, P]
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # [N, P]
            return tf.linalg.matmul(
                K, self.W**2.0,
                transpose_b=True)  # [N, L]  *  [L, P]  ->  [N, P]
