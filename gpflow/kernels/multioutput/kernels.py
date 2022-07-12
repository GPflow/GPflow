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
from typing import Optional, Sequence, Tuple

import tensorflow as tf

from ...base import Parameter, TensorType
from ...experimental.check_shapes import check_shape as cs
from ...experimental.check_shapes import check_shapes, inherit_check_shapes
from ..base import Combination, Kernel


class MultioutputKernel(Kernel):
    """
    Multi Output Kernel class.

    This kernel can represent correlation between outputs of different datapoints.

    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """

    @property
    @abc.abstractmethod
    def num_latent_gps(self) -> int:
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    @abc.abstractmethod
    @check_shapes(
        "X: [batch..., N, D]",
        "X2: [batch2..., N2, D]",
        "return: [batch..., N, P, batch2..., N2, P] if full_output_cov and (X2 is not None)",
        "return: [P, batch..., N, batch2..., N2] if not full_output_cov and (X2 is not None)",
        "return: [batch..., N, P, N, P] if full_output_cov and (X2 is None)",
        "return: [P, batch..., N, N] if not full_output_cov and (X2 is None)",
    )
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param X2: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)]
        """
        raise NotImplementedError

    @abc.abstractmethod
    @check_shapes(
        "X: [batch..., N, D]",
        "return: [batch..., N, P, P] if full_output_cov",
        "return: [batch..., N, P] if not full_output_cov",
    )
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)]
        """
        raise NotImplementedError

    @check_shapes(
        "X: [batch..., N, D]",
        "X2: [batch2..., N2, D]",
        "return: [batch..., N, P, batch2..., N2, P] if full_cov and full_output_cov and (X2 is not None)",
        "return: [P, batch..., N, batch2..., N2] if full_cov and (not full_output_cov) and (X2 is not None)",
        "return: [batch..., N, P, N, P] if full_cov and full_output_cov and (X2 is None)",
        "return: [P, batch..., N, N] if full_cov and (not full_output_cov) and (X2 is None)",
        "return: [batch..., N, P, P] if (not full_cov) and full_output_cov and (X2 is None)",
        "return: [batch..., N, P] if (not full_cov) and (not full_output_cov) and (X2 is None)",
    )
    def __call__(
        self,
        X: TensorType,
        X2: Optional[TensorType] = None,
        *,
        full_cov: bool = False,
        full_output_cov: bool = True,
        presliced: bool = False,
    ) -> tf.Tensor:
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

    .. warning::
       This class is created only for testing and comparison purposes.
       Use `gpflow.kernels` instead for more efficient code.
    """

    def __init__(self, kernel: Kernel, output_dim: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.output_dim = output_dim

    @property
    def num_latent_gps(self) -> int:
        # In this case number of latent GPs (L) == output_dim (P)
        return self.output_dim

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return (self.kernel,)

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        K = self.kernel.K(X, X2)
        rank = tf.rank(X) - 1
        if X2 is None:
            cs(K, "[batch..., N, N]")
            ones = tf.ones((rank,), dtype=tf.int32)
            if full_output_cov:
                multiples = tf.concat([ones, [1, self.output_dim]], 0)
                Ks = cs(tf.tile(K[..., None], multiples), "[batch..., N, N, P]")
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + 1, rank, rank + 2],
                    ],
                    0,
                )
                return cs(tf.transpose(tf.linalg.diag(Ks), perm), "[batch..., N, P, N, P]")
            else:
                multiples = tf.concat([[self.output_dim], ones, [1]], 0)
                return cs(tf.tile(K[None, ...], multiples), "[P, batch..., N, N]")

        else:
            cs(K, "[batch..., N, batch2..., N2]")
            rank2 = tf.rank(X2) - 1
            ones12 = tf.ones((rank + rank2,), dtype=tf.int32)
            if full_output_cov:
                multiples = tf.concat([ones12, [self.output_dim]], 0)
                Ks = cs(tf.tile(K[..., None], multiples), "[batch..., N, batch2..., N2, P]")
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + rank2],
                        rank + tf.range(rank2),
                        [rank + rank2 + 1],
                    ],
                    0,
                )
                return cs(
                    tf.transpose(tf.linalg.diag(Ks), perm), "[batch..., N, P, batch2..., N2, P]"
                )
            else:
                multiples = tf.concat([[self.output_dim], ones12], 0)
                return cs(tf.tile(K[None, ...], multiples), "[P, batch..., N, batch2..., N2]")

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        K = cs(self.kernel.K_diag(X), "[batch..., N]")
        rank = tf.rank(X) - 1
        ones = tf.ones((rank,), dtype=tf.int32)
        multiples = tf.concat([ones, [self.output_dim]], 0)
        Ks = cs(tf.tile(K[..., None], multiples), "[batch..., N, P]")
        return tf.linalg.diag(Ks) if full_output_cov else Ks


class SeparateIndependent(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels: Sequence[Kernel], name: Optional[str] = None) -> None:
        super().__init__(kernels=kernels, name=name)

    @property
    def num_latent_gps(self) -> int:
        return len(self.kernels)

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        rank = tf.rank(X) - 1
        if X2 is None:
            if full_output_cov:
                Kxxs = cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=-1), "[batch..., N, N, P]"
                )
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + 1, rank, rank + 2],
                    ],
                    0,
                )
                return cs(tf.transpose(tf.linalg.diag(Kxxs), perm), "[batch..., N, P, N, P]")
            else:
                return cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=0), "[P, batch..., N, N]"
                )
        else:
            rank2 = tf.rank(X2) - 1
            if full_output_cov:
                Kxxs = cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=-1),
                    "[batch..., N, batch2..., N2, P]",
                )
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + rank2],
                        rank + tf.range(rank2),
                        [rank + rank2 + 1],
                    ],
                    0,
                )
                return cs(
                    tf.transpose(tf.linalg.diag(Kxxs), perm), "[batch..., N, P, batch2..., N2, P]"
                )
            else:
                return cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=0),
                    "[P, batch..., N, batch2..., N2]",
                )

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = False) -> tf.Tensor:
        stacked = cs(tf.stack([k.K_diag(X) for k in self.kernels], axis=-1), "[batch..., N, P]")
        if full_output_cov:
            return cs(tf.linalg.diag(stacked), "[batch..., N, P, P]")
        else:
            return stacked


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
    @check_shapes(
        "X: [batch..., N, D]",
        "X2: [batch2..., N2, D]",
        "return: [L, batch..., N, batch2..., N2]",
    )
    def Kgg(self, X: TensorType, X2: TensorType) -> tf.Tensor:
        raise NotImplementedError


class LinearCoregionalization(IndependentLatent, Combination):
    """
    Linear mixing of the latent GPs to form the output.
    """

    @check_shapes(
        "W: [P, L]",
    )
    def __init__(self, kernels: Sequence[Kernel], W: TensorType, name: Optional[str] = None):
        Combination.__init__(self, kernels=kernels, name=name)
        self.W = Parameter(W)

    @property
    def num_latent_gps(self) -> int:
        return self.W.shape[-1]  # type: ignore[no-any-return]  # L

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    @inherit_check_shapes
    def Kgg(self, X: TensorType, X2: TensorType) -> tf.Tensor:
        return cs(
            tf.stack([k.K(X, X2) for k in self.kernels], axis=0), "[L, batch..., N, batch2..., M]"
        )

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        Kxx = self.Kgg(X, X2)
        if X2 is None:
            cs(Kxx, "[L, batch..., N, N]")
            rank = tf.rank(X) - 1
            ones = tf.ones((rank + 1,), dtype=tf.int32)
            P = tf.shape(self.W)[0]
            L = tf.shape(self.W)[1]
            W_broadcast = cs(
                tf.reshape(self.W, tf.concat([[P, L], ones], 0)), "[P, L, broadcast batch..., 1, 1]"
            )
            KxxW = cs(Kxx[None, ...] * W_broadcast, "[P, L, batch..., N, N]")
            if full_output_cov:
                # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
                WKxxW = cs(tf.tensordot(self.W, KxxW, [[1], [1]]), "[P, P, batch..., N, N]")
                perm = tf.concat(
                    [
                        2 + tf.range(rank),
                        [0, 2 + rank, 1],
                    ],
                    0,
                )
                return cs(tf.transpose(WKxxW, perm), "[batch..., N, P, N, P]")
        else:
            cs(Kxx, "[L, batch..., N, batch2..., N2]")
            rank = tf.rank(X) - 1
            rank2 = tf.rank(X2) - 1
            ones12 = tf.ones((rank + rank2,), dtype=tf.int32)
            P = tf.shape(self.W)[0]
            L = tf.shape(self.W)[1]
            W_broadcast = cs(
                tf.reshape(self.W, tf.concat([[P, L], ones12], 0)),
                "[P, L, broadcast batch..., 1, broadcast batch2..., 1]",
            )
            KxxW = cs(Kxx[None, ...] * W_broadcast, "[P, L, batch..., N, batch2..., N2]")
            if full_output_cov:
                # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
                WKxxW = cs(
                    tf.tensordot(self.W, KxxW, [[1], [1]]), "[P, P, batch..., N, batch2..., N2]"
                )
                perm = tf.concat(
                    [
                        2 + tf.range(rank),
                        [0],
                        2 + rank + tf.range(rank2),
                        [1],
                    ],
                    0,
                )
                return cs(tf.transpose(WKxxW, perm), "[batch..., N, P, batch2..., N2, P]")
        # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
        return tf.reduce_sum(W_broadcast * KxxW, axis=1)

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        K = cs(tf.stack([k.K_diag(X) for k in self.kernels], axis=-1), "[batch..., N, L]")
        rank = tf.rank(X) - 1
        ones = tf.ones((rank,), dtype=tf.int32)

        if full_output_cov:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)
            Wt = cs(tf.transpose(self.W), "[L, P]")
            L = tf.shape(Wt)[0]
            P = tf.shape(Wt)[1]
            return cs(
                tf.reduce_sum(
                    cs(K[..., None, None], "[batch..., N, L, 1, 1]")
                    * cs(tf.reshape(Wt, tf.concat([ones, [L, P, 1]], 0)), "[..., L, P, 1]")
                    * cs(tf.reshape(Wt, tf.concat([ones, [L, 1, P]], 0)), "[..., L, 1, P]"),
                    axis=-3,
                ),
                "[batch..., N, P, P]",
            )
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)
            return cs(tf.linalg.matmul(K, self.W ** 2.0, transpose_b=True), "[batch..., N, P]")
