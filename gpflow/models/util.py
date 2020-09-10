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

from typing import Callable, Union

import numpy as np
import tensorflow as tf

from ..config import default_float
from ..inducing_variables import InducingPoints, InducingVariables
from .model import BayesianModel
from .training_mixins import Data, ExternalDataTrainingLossMixin


def inducingpoint_wrapper(
    inducing_variable: Union[InducingVariables, tf.Tensor, np.ndarray]
) -> InducingVariables:
    """
    This wrapper allows transparently passing either an InducingVariables
    object or an array specifying InducingPoints positions.
    """
    if not isinstance(inducing_variable, InducingVariables):
        inducing_variable = InducingPoints(inducing_variable)
    return inducing_variable


def _assert_equal_data(data1, data2):
    if isinstance(data1, tf.Tensor) and isinstance(data2, tf.Tensor):
        tf.debugging.assert_equal(data1, data2)
    else:
        for v1, v2 in zip(data1, data2):
            tf.debugging.assert_equal(v1, v2)


def training_loss_closure(
    model: BayesianModel, data: Data, **closure_kwargs
) -> Callable[[], tf.Tensor]:
    if isinstance(model, ExternalDataTrainingLossMixin):
        return model.training_loss_closure(data, **closure_kwargs)
    else:
        _assert_equal_data(model.data, data)
        return model.training_loss_closure(**closure_kwargs)


def training_loss(model: BayesianModel, data: Data) -> tf.Tensor:
    if isinstance(model, ExternalDataTrainingLossMixin):
        return model.training_loss(data)
    else:
        _assert_equal_data(model.data, data)
        return model.training_loss()


def maximum_log_likelihood_objective(model: BayesianModel, data: Data) -> tf.Tensor:
    if isinstance(model, ExternalDataTrainingLossMixin):
        return model.maximum_log_likelihood_objective(data)
    else:
        _assert_equal_data(model.data, data)
        return model.maximum_log_likelihood_objective()


def data_input_to_tensor(structure):
    """
    Converts non-tensor elements of a structure to TensorFlow tensors retaining the structure itself.
    The function doesn't keep original element's dtype and forcefully converts
    them to GPflow's default float type.
    """

    def convert_to_tensor(elem):
        if tf.is_tensor(elem):
            return elem
        elif isinstance(elem, np.ndarray):
            return tf.convert_to_tensor(elem)
        return tf.convert_to_tensor(elem, dtype=default_float())

    return tf.nest.map_structure(convert_to_tensor, structure)
