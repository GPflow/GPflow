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

from typing import Any, Callable, Sequence, Union

import numpy as np
import tensorflow as tf

from ..base import AnyNDArray
from ..config import default_float
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingPoints, InducingVariables
from .model import BayesianModel
from .training_mixins import Data, ExternalDataTrainingLossMixin

InducingVariablesLike = Union[InducingVariables, tf.Tensor, AnyNDArray]
InducingPointsLike = Union[InducingPoints, tf.Tensor, AnyNDArray]


def inducingpoint_wrapper(inducing_variable: InducingVariablesLike) -> InducingVariables:
    """
    This wrapper allows transparently passing either an InducingVariables
    object or an array specifying InducingPoints positions.
    """
    if not isinstance(inducing_variable, InducingVariables):
        inducing_variable = InducingPoints(inducing_variable)
    return inducing_variable


def _assert_equal_data(
    data1: Union[tf.Tensor, Sequence[tf.Tensor]], data2: Union[tf.Tensor, Sequence[tf.Tensor]]
) -> None:
    if isinstance(data1, tf.Tensor) and isinstance(data2, tf.Tensor):
        tf.debugging.assert_equal(data1, data2)
    else:
        for v1, v2 in zip(data1, data2):
            tf.debugging.assert_equal(v1, v2)


@check_shapes(
    "data[0]: [N, D]",
    "data[1]: [N, P]",
)
def training_loss_closure(
    model: BayesianModel, data: Data, **closure_kwargs: Any
) -> Callable[[], tf.Tensor]:
    if isinstance(model, ExternalDataTrainingLossMixin):
        return model.training_loss_closure(data, **closure_kwargs)  # type: ignore[no-any-return]
    else:
        _assert_equal_data(model.data, data)
        return model.training_loss_closure(**closure_kwargs)  # type: ignore[no-any-return]


@check_shapes(
    "data[0]: [N, D]",
    "data[1]: [N, P]",
    "return: []",
)
def training_loss(model: BayesianModel, data: Data) -> tf.Tensor:
    if isinstance(model, ExternalDataTrainingLossMixin):
        return model.training_loss(data)
    else:
        _assert_equal_data(model.data, data)
        return model.training_loss()


@check_shapes(
    "data[0]: [N, D]",
    "data[1]: [N, P]",
    "return: []",
)
def maximum_log_likelihood_objective(model: BayesianModel, data: Data) -> tf.Tensor:
    if isinstance(model, ExternalDataTrainingLossMixin):
        return model.maximum_log_likelihood_objective(data)
    else:
        _assert_equal_data(model.data, data)
        return model.maximum_log_likelihood_objective()


def data_input_to_tensor(structure: Any) -> Any:
    """
    Converts non-tensor elements of a structure to TensorFlow tensors retaining the structure
    itself.

    The function doesn't keep original element's dtype and forcefully converts
    them to GPflow's default float type.
    """

    def convert_to_tensor(elem: Any) -> tf.Tensor:
        if tf.is_tensor(elem):
            return elem
        elif isinstance(elem, np.ndarray):
            return tf.convert_to_tensor(elem)
        return tf.convert_to_tensor(elem, dtype=default_float())

    return tf.nest.map_structure(convert_to_tensor, structure)
