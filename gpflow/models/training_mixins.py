"""
This module provides mixin classes to be used in conjunction with inheriting
from gpflow.models.BayesianModel (or its subclass gpflow.models.GPModel).

They provide a unified interface to obtain closures that return the training
loss, to be passed as the first argument to the minimize() method of the
optimizers defined in TensorFlow and GPflow.

All TrainingLossMixin classes assume that self.maximum_a_posteriori_objective()
(which is provided by the BayesianModel base class), will be available. Note
that new models only need to implement the maximum_likelihood_objective method
that is defined as abstract in BayesianModel.

There are different mixins depending on whether the model already contains the
training data (InternalDataTrainingLossMixin), or requires it to be passed in
to the objective function (ExternalDataTrainingLossMixin).
MCMCTrainingLossMixin, a subclass of InternalDataTrainingLossMixin, is provided
for further clarity on what the objective for gradient-based optimization is
for an MCMC model.
"""

import abc
import collections
from typing import Callable, Optional, Tuple, TypeVar, Union

import tensorflow as tf
import numpy as np


InputData = tf.Tensor
OutputData = tf.Tensor
RegressionData = Tuple[InputData, OutputData]
Data = TypeVar("Data", RegressionData, InputData, OutputData)


class InternalDataTrainingLossMixin:
    def training_loss(self) -> tf.Tensor:
        """
        Training loss for models that encapsulate their data.
        """
        return -self.maximum_a_posteriori_objective()

    def training_loss_closure(self, jit=True) -> Callable[[], tf.Tensor]:
        """
        Returns a closure that computes the training loss, which by default is
        wrapped in tf.function(). This can be disabled by passing `jit=False`.

        :param jit: whether to wrap training loss in tf.function()
        """
        if jit:
            return tf.function(self.training_loss)
        return self.training_loss


class ExternalDataTrainingLossMixin:
    def training_loss(self, data: Data) -> tf.Tensor:
        """
        Training loss for models that do not encapsulate the data.
        
        :param data: the data to be used for computing the model objective.
        """
        return -self.maximum_a_posteriori_objective(data)

    def training_loss_closure(
        self, data: Union[Data, collections.abc.Iterator], jit=True
    ) -> Callable[[], tf.Tensor]:
        """
        Returns a closure that computes the training loss, which by default is
        wrapped in tf.function(). This can be disabled by passing `jit=False`.
        
        :param data: the data to be used by the closure for computing the model
            objective. Can be the full dataset or an iterator (e.g.
            `iter(dataset.batch(batch_size))`, where dataset is an instance of
            tf.data.Dataset)
        :param jit: whether to wrap training loss in tf.function()
        """
        training_loss = self.training_loss
        if jit:
            training_loss = tf.function(
                training_loss
            )  # TODO need to add correct input_signature here to allow for differently sized minibatches

        def closure():
            batch = next(data) if isinstance(data, collections.abc.Iterator) else data
            return training_loss(batch)

        return closure


class MCMCTrainingLossMixin(InternalDataTrainingLossMixin):
    def training_loss(self) -> tf.Tensor:
        """
        Training loss for gradient-based relaxation of MCMC models.
        """
        return -self.log_posterior_density()
