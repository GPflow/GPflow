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

from math import sqrt
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf

from .. import logdensities
from ..base import MeanAndVariance, TensorType
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..utilities.parameter_or_function import (
    ConstantOrFunction,
    ParameterOrFunction,
    evaluate_parameter_or_function,
    prepare_parameter_or_function,
)
from .base import ScalarLikelihood
from .utils import inv_probit

DEFAULT_LOWER_BOUND = 1e-6


class Gaussian(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with constant
    variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    def __init__(
        self,
        variance: Optional[ConstantOrFunction] = None,
        *,
        scale: Optional[ConstantOrFunction] = None,
        variance_lower_bound: float = DEFAULT_LOWER_BOUND,
        **kwargs: Any,
    ) -> None:
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``. This is mutually exclusive with `scale`.
        :param scale: The noise scale; must be greater than
            ``sqrt(variance_lower_bound)``. This is mutually exclusive with `variance`.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        self.variance_lower_bound = variance_lower_bound
        self.scale_lower_bound = sqrt(variance_lower_bound)
        if scale is None:
            if variance is None:
                variance = 1.0
            self.variance: Optional[ParameterOrFunction] = prepare_parameter_or_function(
                variance, lower_bound=self.variance_lower_bound
            )
            self.scale: Optional[ParameterOrFunction] = None
        else:
            if variance is None:
                self.variance = None
                self.scale = prepare_parameter_or_function(
                    scale, lower_bound=self.scale_lower_bound
                )
            else:
                assert False, "Cannot set both `variance` and `scale`."

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [broadcast batch..., broadcast N, broadcast P]",
    )
    def _variance(self, X: TensorType) -> tf.Tensor:
        if self.variance is not None:
            return evaluate_parameter_or_function(
                self.variance, X, lower_bound=self.variance_lower_bound
            )
        else:
            assert self.scale is not None  # For mypy.
            return (
                evaluate_parameter_or_function(self.scale, X, lower_bound=self.scale_lower_bound)
                ** 2
            )

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [batch..., N, 1]",
    )
    def variance_at(self, X: TensorType) -> tf.Tensor:
        variance = self._variance(X)
        shape = tf.concat([tf.shape(X)[:-1], [1]], 0)
        return tf.broadcast_to(variance, shape)

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.gaussian(Y, F, self._variance(X))

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:  # pylint: disable=R0201
        return tf.identity(F)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        shape = tf.shape(F)
        return tf.broadcast_to(self._variance(X), shape)

    @inherit_check_shapes
    def _predict_mean_and_var(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType
    ) -> MeanAndVariance:
        return tf.identity(Fmu), Fvar + self._variance(X)

    @inherit_check_shapes
    def _predict_log_density(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + self._variance(X)), axis=-1)

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        variance = self._variance(X)
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / variance,
            axis=-1,
        )


class Exponential(ScalarLikelihood):
    def __init__(self, invlink: Callable[[tf.Tensor], tf.Tensor] = tf.exp, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.invlink = invlink

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.exponential(Y, self.invlink(F))

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self.invlink(F)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return tf.square(self.invlink(F))

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        if self.invlink is tf.exp:
            return tf.reduce_sum(-tf.exp(-Fmu + Fvar / 2) * Y - Fmu, axis=-1)
        return super()._variational_expectations(X, Fmu, Fvar, Y)


class StudentT(ScalarLikelihood):
    def __init__(
        self,
        scale: ConstantOrFunction = 1.0,
        df: float = 3.0,
        scale_lower_bound: float = DEFAULT_LOWER_BOUND,
        **kwargs: Any,
    ) -> None:
        """
        :param scale float: scale parameter
        :param df float: degrees of freedom
        """
        super().__init__(**kwargs)
        self.df = df
        self.scale_lower_bound = scale_lower_bound
        self.scale = prepare_parameter_or_function(scale, lower_bound=self.scale_lower_bound)

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [broadcast batch..., broadcast N, broadcast P]",
    )
    def _scale(self, X: TensorType) -> tf.Tensor:
        return evaluate_parameter_or_function(self.scale, X, lower_bound=self.scale_lower_bound)

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.student_t(Y, F, self._scale(X), self.df)

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return F

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        shape = tf.shape(F)
        var = (self._scale(X) ** 2) * (self.df / (self.df - 2.0))
        return tf.broadcast_to(var, shape)


class Gamma(ScalarLikelihood):
    """
    Use the transformed GP to give the *scale* (inverse rate) of the Gamma
    """

    def __init__(
        self,
        invlink: Callable[[tf.Tensor], tf.Tensor] = tf.exp,
        shape: ConstantOrFunction = 1.0,
        shape_lower_bound: float = DEFAULT_LOWER_BOUND,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.invlink = invlink
        self.shape_lower_bound = shape_lower_bound
        self.shape = prepare_parameter_or_function(shape, lower_bound=self.shape_lower_bound)

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [broadcast batch..., broadcast N, broadcast P]",
    )
    def _shape(self, X: TensorType) -> tf.Tensor:
        return evaluate_parameter_or_function(self.shape, X, lower_bound=self.shape_lower_bound)

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        return logdensities.gamma(Y, self._shape(X), self.invlink(F))

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self._shape(X) * self.invlink(F)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        scale = self.invlink(F)
        return self._shape(X) * (scale ** 2)

    @inherit_check_shapes
    def _variational_expectations(
        self, X: TensorType, Fmu: TensorType, Fvar: TensorType, Y: TensorType
    ) -> tf.Tensor:
        if self.invlink is tf.exp:
            shape = self._shape(X)
            return tf.reduce_sum(
                -shape * Fmu
                - tf.math.lgamma(shape)
                + (shape - 1.0) * tf.math.log(Y)
                - Y * tf.exp(-Fmu + Fvar / 2.0),
                axis=-1,
            )
        else:
            return super()._variational_expectations(X, Fmu, Fvar, Y)


class Beta(ScalarLikelihood):
    """
    This uses a reparameterisation of the Beta density. We have the mean of the
    Beta distribution given by the transformed process:

        m = invlink(f)

    and a scale parameter. The familiar α, β parameters are given by

        m     = α / (α + β)
        scale = α + β

    so:
        α = scale * m
        β  = scale * (1-m)
    """

    def __init__(
        self,
        invlink: Callable[[tf.Tensor], tf.Tensor] = inv_probit,
        scale: ConstantOrFunction = 1.0,
        scale_lower_bound: float = DEFAULT_LOWER_BOUND,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scale_lower_bound = DEFAULT_LOWER_BOUND
        self.scale = prepare_parameter_or_function(scale, lower_bound=self.scale_lower_bound)
        self.invlink = invlink

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [broadcast batch..., broadcast N, broadcast P]",
    )
    def _scale(self, X: TensorType) -> tf.Tensor:
        return evaluate_parameter_or_function(self.scale, X, lower_bound=self.scale_lower_bound)

    @inherit_check_shapes
    def _scalar_log_prob(self, X: TensorType, F: TensorType, Y: TensorType) -> tf.Tensor:
        mean = self.invlink(F)
        scale = self._scale(X)
        alpha = mean * scale
        beta = scale - alpha
        return logdensities.beta(Y, alpha, beta)

    @inherit_check_shapes
    def _conditional_mean(self, X: TensorType, F: TensorType) -> tf.Tensor:
        return self.invlink(F)

    @inherit_check_shapes
    def _conditional_variance(self, X: TensorType, F: TensorType) -> tf.Tensor:
        mean = self.invlink(F)
        var = (mean - tf.square(mean)) / (self._scale(X) + 1.0)
        shape = tf.shape(F)
        return tf.broadcast_to(var, shape)
