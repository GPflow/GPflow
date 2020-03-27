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
MCMCTrainingLossMixin, a subclass of InternalDataTrainingLossMixin, is provided
for further clarity on what the objective for gradient-based optimization is
for an MCMC model.
"""

import abc
from typing import Callable, Iterator, Optional, Tuple, TypeVar, Union

import tensorflow as tf
from tensorflow.python.data.ops.iterator_ops import OwnedIterator as DatasetOwnedIterator
import numpy as np


InputData = tf.Tensor
OutputData = tf.Tensor
RegressionData = Tuple[InputData, OutputData]
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

    def training_loss(self) -> tf.Tensor:
        """
        Returns the training loss for this model.
        """
        return self._training_loss()

    def training_loss_closure(self, jit=True) -> Callable[[], tf.Tensor]:
        """
        Convenience method. Returns a closure which itself returns the training loss. This closure
        can be passed to the minimize methods on :class:`gpflow.optimizers.Scipy` and subclasses of
        `tf.optimizers.Optimizer`.

        :param jit: If `True` (default), compile the training loss function in a TensorFlow graph
            by wrapping it in tf.function()
        """
        if jit:
            return tf.function(self.training_loss)
        return self.training_loss


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

    def training_loss(self, data: Data) -> tf.Tensor:
        """
        Returns the training loss for this model.
        
        :param data: the data to be used for computing the model objective.
        """
        return self._training_loss(data)

    def training_loss_closure(
        self, data: Union[Data, Iterator[Data]], jit=True, input_signature=None,
    ) -> Callable[[], tf.Tensor]:
        """
        Returns a closure that computes the training loss, which by default is
        wrapped in tf.function(). This can be disabled by passing `jit=False`.
        
        :param data: the data to be used by the closure for computing the model
            objective. Can be the full dataset or an iterator (e.g.
            `iter(dataset.batch(batch_size))`, where dataset is an instance of
            tf.data.Dataset)
        :param jit: whether to wrap training loss in tf.function()
        :param input_signature: to be passed to tf.function() when jit=True;
            will be inferred from the iterator if `data` is an iterator on a
            tf.data.Dataset.
        """
        training_loss = self.training_loss
        if jit:
            if isinstance(data, DatasetOwnedIterator):
                input_signature = [data.element_spec]
            training_loss = tf.function(training_loss, input_signature=input_signature)

        def closure():
            batch = next(data) if isinstance(data, Iterator) else data
            return training_loss(batch)

        return closure


class MCMCTrainingLossMixin(InternalDataTrainingLossMixin):
    def training_loss(self) -> tf.Tensor:
        """
        Training loss for gradient-based relaxation of MCMC models.
        """
        return -self.log_posterior_density()