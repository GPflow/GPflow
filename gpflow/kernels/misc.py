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

from ..base import Parameter
from ..utilities import positive, to_default_float
from .base import ActiveDims, Kernel


class ArcCosine(Kernel):
    """
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0. The key reference is

    ::

        @incollection{NIPS2009_3628,
            title = {Kernel Methods for Deep Learning},
            author = {Youngmin Cho and Lawrence K. Saul},
            booktitle = {Advances in Neural Information Processing Systems 22},
            year = {2009},
            url = {http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf}
        }
    """

    implemented_orders = {0, 1, 2}

    def __init__(
        self,
        order: int = 0,
        variance=1.0,
        weight_variances=1.0,
        bias_variance=1.0,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ):
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
        return self.weight_variances.shape.ndims > 0

    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return tf.reduce_sum(self.weight_variances * tf.square(X), axis=1) + self.bias_variance
        return (
            tf.linalg.matmul((self.weight_variances * X), X2, transpose_b=True) + self.bias_variance
        )

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
        elif self.order == 2:
            return 3.0 * tf.sin(theta) * tf.cos(theta) + (np.pi - theta) * (
                1.0 + 2.0 * tf.cos(theta) ** 2
            )

    def K(self, X, X2=None):
        X_denominator = tf.sqrt(self._weighted_product(X))
        if X2 is None:
            X2 = X
            X2_denominator = X_denominator
        else:
            X2_denominator = tf.sqrt(self._weighted_product(X2))

        numerator = self._weighted_product(X, X2)
        cos_theta = numerator / X_denominator[:, None] / X2_denominator[None, :]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return (
            self.variance
            * (1.0 / np.pi)
            * self._J(theta)
            * X_denominator[:, None] ** self.order
            * X2_denominator[None, :] ** self.order
        )

    def K_diag(self, X):
        X_product = self._weighted_product(X)
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
    ):
        """
        :param output_dim: number of outputs expected (0 <= X < output_dim)
        :param rank: number of degrees of correlation between outputs
        """

        # assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(active_dims=active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        W = 0.1 * np.ones((self.output_dim, self.rank))
        kappa = np.ones(self.output_dim)
        self.W = Parameter(W)
        self.kappa = Parameter(kappa, transform=positive())

    def output_covariance(self):
        B = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(self.kappa)
        return B

    def output_variance(self):
        B_diag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return B_diag

    def K(self, X, X2=None):
        shape_constraints = [
            (X, [..., "N", 1]),
        ]
        if X2 is not None:
            shape_constraints.append((X2, [..., "M", 1]))
        tf.debugging.assert_shapes(shape_constraints)

        X = tf.cast(X[..., 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[..., 0], tf.int32)

        B = self.output_covariance()
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def K_diag(self, X):
        tf.debugging.assert_shapes([(X, [..., "N", 1])])
        X = tf.cast(X[..., 0], tf.int32)
        B_diag = self.output_variance()
        return tf.gather(B_diag, X)
