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

import abc
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from deprecated import deprecated

from ..base import Module, Parameter, TensorData, TensorType
from ..experimental.check_shapes import ErrorContext, Shape, check_shapes, register_get_shape
from ..utilities import positive


class InducingVariables(Module, abc.ABC):
    """
    Abstract base class for inducing variables.
    """

    @property
    @abc.abstractmethod
    def num_inducing(self) -> tf.Tensor:
        """
        Returns the number of inducing variables, relevant for example to determine the size of the
        variational distribution.
        """
        raise NotImplementedError

    @deprecated(
        reason="len(iv) should return an `int`, but this actually returns a `tf.Tensor`."
        " Use `iv.num_inducing` instead."
    )
    def __len__(self) -> tf.Tensor:
        return self.num_inducing

    @property
    @abc.abstractmethod
    def shape(self) -> Shape:
        """
        Return the shape of these inducing variables.

        Shape should be some variation of ``[M, D, P]``, where:

        * ``M`` is the number of inducing variables.
        * ``D`` is the number of input dimensions.
        * ``P`` is the number of output dimensions (1 if this is not a multi-output inducing
          variable).
        """


@register_get_shape(InducingVariables)
def get_scalar_shape(shaped: InducingVariables, context: ErrorContext) -> Shape:
    return shaped.shape


class InducingPointsBase(InducingVariables):
    @check_shapes(
        "Z: [M, D]",
    )
    def __init__(self, Z: TensorData, name: Optional[str] = None):
        """
        :param Z: The initial positions of the inducing points.
        """
        super().__init__(name=name)
        if not isinstance(Z, (tf.Variable, tfp.util.TransformedVariable)):
            Z = Parameter(Z)
        self.Z = Z

    @property  # type: ignore[misc]  # mypy doesn't like decorated properties.
    @check_shapes(
        "return: []",
    )
    def num_inducing(self) -> Optional[tf.Tensor]:
        return tf.shape(self.Z)[0]

    @property
    def shape(self) -> Shape:
        shape = self.Z.shape
        if not shape:
            return None
        return tuple(shape) + (1,)


class InducingPoints(InducingPointsBase):
    """
    Real-space inducing points
    """


class Multiscale(InducingPointsBase):
    r"""
    Multi-scale inducing variables

    Originally proposed in :cite:t:`NIPS2009_3876`.
    """

    @check_shapes(
        "Z: [M, D]",
        "scales: [M, D]",
    )
    def __init__(self, Z: TensorData, scales: TensorData):
        super().__init__(Z)
        # Multi-scale inducing_variable widths (std. dev. of Gaussian)
        self.scales = Parameter(scales, transform=positive())

    @staticmethod
    @check_shapes(
        "A: [N, D]",
        "B: [M, D]",
        "sc: [broadcast N, broadcast M, D]",
        "return: [N, M]",
    )
    def _cust_square_dist(A: TensorType, B: TensorType, sc: TensorType) -> tf.Tensor:
        """
        Custom version of _square_dist that allows sc to provide per-datapoint length
        scales.
        """
        return tf.reduce_sum(tf.square((tf.expand_dims(A, 1) - tf.expand_dims(B, 0)) / sc), 2)
