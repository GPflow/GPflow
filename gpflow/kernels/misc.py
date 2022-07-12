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

from typing import Optional

import numpy as np
import tensorflow as tf

from ..base import AnyNDArray, Parameter, TensorType
from ..experimental.check_shapes import check_shape as cs
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..utilities import positive, to_default_float
from .base import ActiveDims, Kernel


class ArcCosine(Kernel):
    """
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0.

    The key reference is :cite:t:`NIPS2009_3628`.
    """

    implemented_orders = {0, 1, 2}

    @check_shapes(
        "variance: []",
        "weight_variances: [broadcast n_active_dims]",
        "bias_variance: []",
    )
    def __init__(
        self,
        order: int = 0,
        variance: TensorType = 1.0,
        weight_variances: TensorType = 1.0,
        bias_variance: TensorType = 1.0,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        :param order: specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order
        :param variance: the (initial) value for the variance parameter
        :param weight_variances: the (initial) value for the weight_variances parameter,
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param bias_variance: the (initial) value for the bias_variance parameter
            defaults to 1.0
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(active_dims=active_dims, name=name)

        if order not in self.implemented_orders:
            raise ValueError("Requested kernel order is not implemented.")
        self.order = order

        self.variance = Parameter(variance, transform=positive())
        self.bias_variance = Parameter(bias_variance, transform=positive())
        self.weight_variances = Parameter(weight_variances, transform=positive())
        self._validate_ard_active_dims(self.weight_variances)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        ndims: int = self.weight_variances.shape.ndims
        return ndims > 0

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [batch..., N]",
    )
    def _diag_weighted_product(self, X: TensorType) -> tf.Tensor:
        return tf.reduce_sum(self.weight_variances * tf.square(X), axis=-1) + self.bias_variance

    @check_shapes(
        "X: [batch..., N, D]",
        "X2: [batch2..., N2, D]",
        "return: [batch..., N, batch2..., N2] if X2 is not None",
        "return: [batch..., N, N] if X2 is None",
    )
    def _full_weighted_product(self, X: TensorType, X2: Optional[TensorType]) -> tf.Tensor:
        if X2 is None:
            return (
                tf.linalg.matmul((self.weight_variances * X), X, transpose_b=True)
                + self.bias_variance
            )

        else:
            D = tf.shape(X)[-1]
            N = tf.shape(X)[-2]
            N2 = tf.shape(X2)[-2]
            batch = tf.shape(X)[:-2]
            batch2 = tf.shape(X2)[:-2]
            rank = tf.rank(X) - 2
            rank2 = tf.rank(X2) - 2
            ones = tf.ones((rank,), tf.int32)
            ones2 = tf.ones((rank2,), tf.int32)

            X = cs(
                tf.reshape(X, tf.concat([batch, ones2, [N, D]], 0)),
                "[batch..., broadcast batch2..., N, D]",
            )
            X2 = cs(
                tf.reshape(X2, tf.concat([ones, batch2, [N2, D]], 0)),
                "[broadcast batch..., batch2..., N2, D]",
            )
            result = cs(
                tf.linalg.matmul((self.weight_variances * X), X2, transpose_b=True)
                + self.bias_variance,
                "[batch..., batch2..., N, N2]",
            )

            indices = tf.concat(
                [
                    tf.range(rank),
                    [rank + rank2],
                    tf.range(rank2) + rank,
                    [rank + rank2 + 1],
                ],
                axis=0,
            )
            return tf.transpose(result, indices)

    @check_shapes(
        "theta: [any...]",
        "return: [any...]",
    )
    def _J(self, theta: TensorType) -> TensorType:
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
        else:
            assert self.order == 2, f"Don't know how to handle order {self.order}."
            return 3.0 * tf.sin(theta) * tf.cos(theta) + (np.pi - theta) * (
                1.0 + 2.0 * tf.cos(theta) ** 2
            )

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        X_denominator = cs(tf.sqrt(self._diag_weighted_product(X)), "[batch..., N]")
        if X2 is None:
            X2_denominator = cs(X_denominator[..., None, :], "[batch..., 1, N]")
            X_denominator = cs(X_denominator[..., :, None], "[batch..., N, 1]")
            numerator = cs(self._full_weighted_product(X, None), "[batch..., N, N]")
        else:
            X2_denominator = cs(tf.sqrt(self._diag_weighted_product(X2)), "[batch2..., N2]")

            batch = tf.shape(X)[:-1]
            batch2 = tf.shape(X2)[:-1]
            ones = tf.ones((tf.rank(X) - 1,), tf.int32)
            ones2 = tf.ones((tf.rank(X2) - 1,), tf.int32)

            X_denominator = cs(
                tf.reshape(X_denominator, tf.concat([batch, ones2], 0)),
                "[batch..., N, broadcast batch2..., 1]",
            )
            X2_denominator = cs(
                tf.reshape(X2_denominator, tf.concat([ones, batch2], 0)),
                "[broadcast batch..., 1, batch2..., N2]",
            )
            numerator = cs(self._full_weighted_product(X, X2), "[batch..., N, batch2..., N2]")

        cos_theta = numerator / X_denominator / X2_denominator
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return (
            self.variance
            * (1.0 / np.pi)
            * self._J(theta)
            * X_denominator ** self.order
            * X2_denominator ** self.order
        )

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        X_product = self._diag_weighted_product(X)
        const = (1.0 / np.pi) * self._J(to_default_float(0.0))
        return self.variance * const * X_product ** self.order


class Coregion(Kernel):
    """
    A Coregionalization kernel. The inputs to this kernel are _integers_ (we
    cast them from floats as needed) which usually specify the *outputs* of a
    Coregionalization model.

    The kernel function is an indexing of a positive-definite matrix:

      K(x, y) = B[x, y] .

    To ensure that B is positive-definite, it is specified by the two
    parameters of this kernel, W and kappa:

      B = W Wᵀ + diag(kappa) .

    We refer to the size of B as "output_dim x output_dim", since this is the
    number of outputs in a coregionalization model. We refer to the number of
    columns on W as 'rank': it is the number of degrees of correlation between
    the outputs.

    NB. There is a symmetry between the elements of W, which creates a local
    minimum at W=0. To avoid this, it is recommended to initialize the
    optimization (or MCMC chain) using a random W.
    """

    def __init__(
        self,
        output_dim: int,
        rank: int,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        :param output_dim: number of outputs expected (0 <= X < output_dim)
        :param rank: number of degrees of correlation between outputs
        """

        # assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(active_dims=active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        W: AnyNDArray = 0.1 * np.ones((self.output_dim, self.rank))
        kappa = np.ones(self.output_dim)
        self.W = Parameter(W)
        self.kappa = Parameter(kappa, transform=positive())

    @check_shapes(
        "return: [P, P]",
    )
    def output_covariance(self) -> tf.Tensor:
        B = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(self.kappa)
        return B

    @check_shapes(
        "return: [P]",
    )
    def output_variance(self) -> tf.Tensor:
        B_diag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return B_diag

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        cs(X, "[batch..., N, 1]  # The `Coregion` kernel requires a 1D input space.")

        B = cs(self.output_covariance(), "[O, O]")
        X = cs(tf.cast(X[..., 0], tf.int32), "[batch..., N]")
        if X2 is None:
            batch = tf.shape(X)[:-1]
            N = tf.shape(X)[-1]
            O = tf.shape(B)[-1]

            result = cs(tf.gather(B, X), "[batch..., N, O]")
            result = cs(tf.reshape(result, [-1, N, O]), "[flat_batch, N, O]")
            flat_X = cs(tf.reshape(X, [-1, N]), "[flat_batch, N]")
            result = cs(tf.gather(result, flat_X, axis=2, batch_dims=1), "[flat_batch, N, N]")
            result = cs(tf.reshape(result, tf.concat([batch, [N, N]], 0)), "[batch..., N, N]")
        else:
            X2 = cs(tf.cast(X2[..., 0], tf.int32), "[batch2..., N2]")

            rank2 = tf.rank(X2)

            result = cs(tf.gather(B, X2), "[batch2..., N2, O]")
            result = cs(
                tf.transpose(result, tf.concat([[rank2], tf.range(rank2)], 0)), "[O, batch2..., N2]"
            )
            result = cs(tf.gather(result, X), "[batch..., N, batch2..., N2]")

        return result

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        cs(X, "[batch..., N, 1]  # The `Coregion` kernel requires a 1D input space.")

        X = tf.cast(X[..., 0], tf.int32)
        B_diag = self.output_variance()
        return tf.gather(B_diag, X)
