# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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

"""
This module provides mixin classes to be used in conjunction with inheriting
from gpflow.models.BayesianModel (or its subclass gpflow.models.GPModel).

They provide a unified interface to obtain closures that return the training
loss, to be passed as the first argument to the minimize() method of the
optimizers defined in TensorFlow and GPflow.

All TrainingLossMixin classes assume that self._training_loss()
(which is provided by the BayesianModel base class), will be available. Note
that new models only need to implement the maximum_log_likelihood_objective
method that is defined as abstract in BayesianModel.

There are different mixins depending on whether the model already contains the
training data (InternalDataTrainingLossMixin), or requires it to be passed in
to the objective function (ExternalDataTrainingLossMixin).
"""
from typing import Callable, TypeVar, Union

import tensorflow as tf
from tensorflow.python.data.ops.iterator_ops import OwnedIterator as DatasetOwnedIterator

from ..base import InputData, OutputData, RegressionData
from ..experimental.check_shapes import check_shapes

Data = TypeVar("Data", RegressionData, InputData, OutputData)


class InternalDataTrainingLossMixin:
    """
    Mixin utility for training loss methods for models that own their own data. It provides

      - a uniform API for the training loss :meth:`training_loss`
      - a convenience method :meth:`training_loss_closure` for constructing the closure expected by
        various optimizers, namely :class:`gpflow.optimizers.Scipy` and subclasses of
        `tf.optimizers.Optimizer`.

    See :class:`ExternalDataTrainingLossMixin` for an equivalent mixin for models that do **not**
    own their own data.
    """

    @check_shapes(
        "return: []",
    )
    def training_loss(self) -> tf.Tensor:
        """
        Returns the training loss for this model.
        """
        # Type-ignore is because _training_loss should be added by implementing class.
        return self._training_loss()  # type: ignore[attr-defined]

    def training_loss_closure(self, *, compile: bool = True) -> Callable[[], tf.Tensor]:
        """
        Convenience method. Returns a closure which itself returns the training loss. This closure
        can be passed to the minimize methods on :class:`gpflow.optimizers.Scipy` and subclasses of
        `tf.optimizers.Optimizer`.

        :param compile: If `True` (default), compile the training loss function in a TensorFlow
            graph by wrapping it in tf.function()
        """
        closure = self.training_loss
        if compile:
            closure = tf.function(closure)
        return closure


class ExternalDataTrainingLossMixin:
    """
    Mixin utility for training loss methods for models that do **not** own their own data.
    It provides

      - a uniform API for the training loss :meth:`training_loss`
      - a convenience method :meth:`training_loss_closure` for constructing the closure expected by
        various optimizers, namely :class:`gpflow.optimizers.Scipy` and subclasses of
        `tf.optimizers.Optimizer`.

    See :class:`InternalDataTrainingLossMixin` for an equivalent mixin for models that **do** own
    their own data.
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "return: []",
    )
    def training_loss(self, data: Data) -> tf.Tensor:
        """
        Returns the training loss for this model.

        :param data: the data to be used for computing the model objective.
        """
        # Type-ignore is because _training_loss should be added by implementing class.
        return self._training_loss(data)  # type: ignore[attr-defined]

    def training_loss_closure(
        self,
        data: Union[Data, DatasetOwnedIterator],
        *,
        compile: bool = True,
    ) -> Callable[[], tf.Tensor]:
        """
        Returns a closure that computes the training loss, which by default is
        wrapped in tf.function(). This can be disabled by passing `compile=False`.

        :param data: the data to be used by the closure for computing the model
            objective. Can be the full dataset or an iterator, e.g.
            `iter(dataset.batch(batch_size))`, where dataset is an instance of
            tf.data.Dataset.
        :param compile: if True, wrap training loss in tf.function()
        """
        training_loss = self.training_loss

        if isinstance(data, DatasetOwnedIterator):
            if compile:
                # lambda because: https://github.com/GPflow/GPflow/issues/1929
                training_loss_lambda = lambda d: self.training_loss(d)
                input_signature = [data.element_spec]
                training_loss = tf.function(training_loss_lambda, input_signature=input_signature)

            def closure() -> tf.Tensor:
                assert isinstance(data, DatasetOwnedIterator)  # Hint for mypy.
                batch = next(data)
                return training_loss(batch)

        else:

            def closure() -> tf.Tensor:
                return training_loss(data)

            if compile:
                closure = tf.function(closure)

        return closure
