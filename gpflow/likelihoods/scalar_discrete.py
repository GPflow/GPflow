# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Any, Callable

import numpy as np
import tensorflow as tf

from .. import logdensities
from ..base import AnyNDArray, MeanAndVariance, Parameter, TensorType
from ..config import default_float
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..utilities import positive, to_default_int
from .base import ScalarLikelihood
from .utils import inv_probit


class Poisson(ScalarLikelihood):
    r"""
    Poisson likelihood for use with count data, where the rate is given by the (transformed) GP.

    let g(.) be the inverse-link function, then this likelihood represents

    p(yᵢ | fᵢ) = Poisson(yᵢ | g(fᵢ) * binsize)

    Note:binsize
    For use in a Log Gaussian Cox process (doubly stochastic model) where the
    rate function of an inhomogeneous Poisson process is given by a GP.  The
    intractable likelihood can be approximated via a Riemann sum (with bins
    of size 'binsize') and using this Poisson likelihood.
    """

    def __init__(
        self,
        invlink: Callable[[tf.Tensor], tf.Tensor] = tf.exp,
        binsize: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.invlink = invlink
        self.binsize: AnyNDArray = np.array(binsize, dtype=default_float())

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.poisson(Y, self.invlink(F) * self.binsize)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self.invlink(F) * self.binsize

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self.invlink(F) * self.binsize

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        if self.invlink is tf.exp:
            return tf.reduce_sum(
                Y * Fmu
                - tf.exp(Fmu + Fvar / 2) * self.binsize
                - tf.math.lgamma(Y + 1)
                + Y * tf.math.log(self.binsize),
                axis=-1,
            )
        return super()._variational_expectations(X, Fmu, Fvar, Y)


class Bernoulli(ScalarLikelihood):
    def __init__(
        self, invlink: Callable[[tf.Tensor], tf.Tensor] = inv_probit, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.invlink = invlink

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.bernoulli(Y, self.invlink(F))

    @inherit_check_shapes
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return super()._predict_mean_and_var(X, Fmu, Fvar)

    @inherit_check_shapes
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        p = self.predict_mean_and_var(X, Fmu, Fvar)[0]
        return tf.reduce_sum(logdensities.bernoulli(Y, p), axis=-1)

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self.invlink(F)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        p = self.conditional_mean(X, F)
        return p - (p ** 2)


class Ordinal(ScalarLikelihood):
    """
    A likelihood for doing ordinal regression.

    The data are integer values from 0 to k, and the user must specify (k-1)
    'bin edges' which define the points at which the labels switch. Let the bin
    edges be [a₀, a₁, ... aₖ₋₁], then the likelihood is

    p(Y=0|F) = ɸ((a₀ - F) / σ)
    p(Y=1|F) = ɸ((a₁ - F) / σ) - ɸ((a₀ - F) / σ)
    p(Y=2|F) = ɸ((a₂ - F) / σ) - ɸ((a₁ - F) / σ)
    ...
    p(Y=K|F) = 1 - ɸ((aₖ₋₁ - F) / σ)

    where ɸ is the cumulative density function of a Gaussian (the inverse probit
    function) and σ is a parameter to be learned.

    A reference is :cite:t:`chu2005gaussian`.
    """

    @check_shapes(
        "bin_edges: [num_bins_minus_1]",
    )
    def __init__(self, bin_edges: AnyNDArray, **kwargs: Any) -> None:
        """
        bin_edges is a numpy array specifying at which function value the
        output label should switch. If the possible Y values are 0...K, then
        the size of bin_edges should be (K-1).
        """
        super().__init__(**kwargs)
        self.bin_edges = bin_edges
        self.num_bins = bin_edges.size + 1
        self.sigma = Parameter(1.0, transform=positive())

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        Y = to_default_int(Y)
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        selected_bins_left = tf.gather(scaled_bins_left, Y)
        selected_bins_right = tf.gather(scaled_bins_right, Y)

        return tf.math.log(
            inv_probit(selected_bins_left - F / self.sigma)
            - inv_probit(selected_bins_right - F / self.sigma)
            + 1e-6
        )

    @check_shapes(
        "F: [batch..., latent_dim]",
        "return: [batch_and_latent_dim, num_bins]",
    )
    def _make_phi(self, F: TensorType) -> tf.Tensor:
        """
        A helper function for making predictions. Constructs a probability
        matrix where each row output the probability of the corresponding
        label, and the rows match the entries of F.

        Note that a matrix of F values is flattened.
        """
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        return inv_probit(scaled_bins_left - tf.reshape(F, (-1, 1)) / self.sigma) - inv_probit(
            scaled_bins_right - tf.reshape(F, (-1, 1)) / self.sigma
        )

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=default_float()), (-1, 1))
        return tf.reshape(tf.linalg.matmul(phi, Ys), tf.shape(F))

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=default_float()), (-1, 1))
        E_y = phi @ Ys
        E_y2 = phi @ (Ys ** 2)
        return tf.reshape(E_y2 - E_y ** 2, tf.shape(F))
