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

from ..base import Module, Parameter, TensorData, TensorType
from ..config import default_float
from ..utilities import positive


class InducingVariables(Module):
    """
    Abstract base class for inducing variables.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of inducing variables, relevant for example
        to determine the size of the variational distribution.
        """
        raise NotImplementedError

    @property
    def num_inducing(self) -> int:
        return self.__len__()


class InducingPointsBase(InducingVariables):
    def __init__(self, Z: TensorData, name: Optional[str] = None):
        """
        :param Z: the initial positions of the inducing points, size [M, D]
        """
        super().__init__(name=name)
        if not isinstance(Z, (tf.Variable, tfp.util.TransformedVariable)):
            Z = Parameter(Z)
        self.Z = Z

    def __len__(self) -> int:
        return tf.shape(self.Z)[0]


class InducingPoints(InducingPointsBase):
    """
    Real-space inducing points
    """


class Multiscale(InducingPointsBase):
    r"""
    Multi-scale inducing variables

    Originally proposed in

    ::

      @incollection{NIPS2009_3876,
        title = {Inter-domain Gaussian Processes for Sparse Inference using Inducing Features},
        author = {Miguel L\'{a}zaro-Gredilla and An\'{\i}bal Figueiras-Vidal},
        booktitle = {Advances in Neural Information Processing Systems 22},
        year = {2009},
      }
    """

    def __init__(self, Z: TensorData, scales: TensorData):
        super().__init__(Z)
        # Multi-scale inducing_variable widths (std. dev. of Gaussian)
        self.scales = Parameter(scales, transform=positive())
        if self.Z.shape != self.scales.shape:
            raise ValueError(
                "Input locations `Z` and `scales` must have the same shape."
            )  # pragma: no cover

    @staticmethod
    def _cust_square_dist(A: TensorType, B: TensorType, sc: TensorType) -> tf.Tensor:
        """
        Custom version of _square_dist that allows sc to provide per-datapoint length
        scales. sc: [N, M, D].
        """
        return tf.reduce_sum(tf.square((tf.expand_dims(A, 1) - tf.expand_dims(B, 0)) / sc), 2)
