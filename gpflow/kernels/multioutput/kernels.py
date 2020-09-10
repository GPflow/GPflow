# Copyright 2018-2020 The GPflow Contributors. All Rights Reserved.
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

from ...base import Parameter
from ..base import Combination, Kernel


class MultioutputKernel(Kernel):
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

    @property
    @abc.abstractmethod
    def num_latent_gps(self):
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    @abc.abstractmethod
    def K(self, X, X2=None, full_output_cov=True):
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.
        :param X: data matrix, [N1, D]
        :param X2: data matrix, [N2, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)] with shape
        - [N1, P, N2, P] if `full_output_cov` = True
        - [P, N1, N2] if `full_output_cov` = False
        """
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, X, full_output_cov=True):
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.
        :param X: data matrix, [N, D]
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)] with shape
        - [N, P, N, P] if `full_output_cov` = True
        - [N, P] if `full_output_cov` = False
        """
        raise NotImplementedError

    def __call__(self, X, X2=None, *, full_cov=False, full_output_cov=True, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, X2, full_output_cov=full_output_cov)


class SharedIndependent(MultioutputKernel):
    """
    - Shared: we use the same kernel for each latent GP
    - Independent: Latents are uncorrelated a priori.
    Note: this class is created only for testing and comparison purposes.
    Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(self, kernel: Kernel, output_dim: int):
        super().__init__()
        self.kernel = kernel
        self.output_dim = output_dim

    @property
    def num_latent_gps(self):
        # In this case number of latent GPs (L) == output_dim (P)
        return self.output_dim

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return (self.kernel,)

    def K(self, X, X2=None, full_output_cov=True):
        K = self.kernel.K(X, X2)  # [N, N2]
        if full_output_cov:
            Ks = tf.tile(K[..., None], [1, 1, self.output_dim])  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Ks), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.tile(K[None, ...], [self.output_dim, 1, 1])  # [P, N, N2]

    def K_diag(self, X, full_output_cov=True):
        K = self.kernel.K_diag(X)  # N
        Ks = tf.tile(K[:, None], [1, self.output_dim])  # [N, P]
        return tf.linalg.diag(Ks) if full_output_cov else Ks  # [N, P, P] or [N, P]


class SeparateIndependent(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels, name=None):
        super().__init__(kernels=kernels, name=name)

    @property
    def num_latent_gps(self):
        return len(self.kernels)

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def K(self, X, X2=None, full_output_cov=True):
        if full_output_cov:
            Kxxs = tf.stack([k.K(X, X2) for k in self.kernels], axis=2)  # [N, N2, P]
            return tf.transpose(tf.linalg.diag(Kxxs), [0, 2, 1, 3])  # [N, P, N2, P]
        else:
            return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [P, N, N2]

    def K_diag(self, X, full_output_cov=False):
        stacked = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, P]
        return tf.linalg.diag(stacked) if full_output_cov else stacked  # [N, P, P]  or  [N, P]


class IndependentLatent(MultioutputKernel):
    """
    Base class for multioutput kernels that are constructed from independent
    latent Gaussian processes.

    It should always be possible to specify inducing variables for such kernels
    that give a block-diagonal Kuu, which can be represented as a [L, M, M]
    tensor. A reasonable (but not optimal) inference procedure can be specified
    by placing the inducing points in the latent processes and simply computing
    Kuu [L, M, M] and Kuf [N, P, M, L] and using `fallback_independent_latent_
    conditional()`. This can be specified by using `Fallback{Separate|Shared}
    IndependentInducingVariables`.
    """

    @abc.abstractmethod
    def Kgg(self, X, X2):
        raise NotImplementedError


class LinearCoregionalization(IndependentLatent, Combination):
    """
    Linear mixing of the latent GPs to form the output.
    """

    def __init__(self, kernels, W, name=None):
        Combination.__init__(self, kernels=kernels, name=name)
        self.W = Parameter(W)  # [P, L]

    @property
    def num_latent_gps(self):
        return self.W.shape[-1]  # L

    @property
    def latent_kernels(self):
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    def Kgg(self, X, X2):
        return tf.stack([k.K(X, X2) for k in self.kernels], axis=0)  # [L, N, N2]

    def K(self, X, X2=None, full_output_cov=True):
        Kxx = self.Kgg(X, X2)  # [L, N, N2]
        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None]  # [P, L, N, N2]
        if full_output_cov:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # [P, P, N, N2]
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # [N, P, N2, P]
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
            return tf.reduce_sum(self.W[:, :, None, None] * KxxW, [1])  # [P, N, N2]

    def K_diag(self, X, full_output_cov=True):
        K = tf.stack([k.K_diag(X) for k in self.kernels], axis=1)  # [N, L]
        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # [N, P, P]
            Wt = tf.transpose(self.W)  # [L, P]
            return tf.reduce_sum(
                K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :], axis=1
            )  # [N, P, P]
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # [N, P]
            return tf.linalg.matmul(
                K, self.W ** 2.0, transpose_b=True
            )  # [N, L]  *  [L, P]  ->  [N, P]
