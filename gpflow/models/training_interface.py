import abc
import collections
from typing import Callable, Optional, Tuple, TypeVar


import tensorflow as tf
import numpy as np

InputData = tf.Tensor
OutputData = tf.Tensor
RegressionData = Tuple[InputData, OutputData]
Data = TypeVar("Data", RegressionData, InputData)


class InternalDataTrainingLossMixin:
    def training_loss(self):
        """
        Specify a training loss for this model
        """
        return -self.maximum_a_posteriori_objective(data)

    def training_loss_closure(self, jit=True) -> Callable[[], tf.Tensor]:
        if jit:
            return tf.function(self.training_loss)
        return self.training_loss


class ExternalDataTrainingLossMixin:
    def training_loss(self, data):
        """
        Specify a training loss for this model
        """
        return -self.maximum_a_posteriori_objective(data)

    def training_loss_closure(self, data: Data, jit=True) -> Callable[[], tf.Tensor]:
        training_loss = self.training_loss
        if jit:
            training_loss = tf.function(
                self.training_loss
            )  # TODO need to add correct input_signature here to allow for differently sized minibatches

        def closure():
            batch = next(data) if isinstance(data, collections.abc.Iterator) else data
            return training_loss(batch)

        return closure
