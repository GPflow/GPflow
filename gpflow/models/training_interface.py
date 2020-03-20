import abc
import collections
from typing import Callable, Optional, Tuple, TypeVar


import tensorflow as tf
import numpy as np

InputData = tf.Tensor
OutputData = tf.Tensor
RegressionData = Tuple[InputData, OutputData]
Data = TypeVar("Data", RegressionData, InputData)


class TrainingInterface:
    @abc.abstractmethod
    def training_loss(self, *args, **kwargs) -> tf.Tensor:
        """
        Specify a training loss for this model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def training_loss_closure(self, *args, **kwargs):
        """
        Return a closure with a training loss for this module
        """
        raise NotImplementedError


class InternalDataTrainingInterface(TrainingInterface):
    def training_loss(self):
        """
        Specify a training loss for this model
        """
        raise NotImplementedError

    def training_loss_closure(self, jit=True) -> Callable[[], tf.Tensor]:
        if jit:
            return tf.function(self.training_loss)
        return self.training_loss


class ExternalDataTrainingInterface(TrainingInterface):
    def training_loss(self, data):
        """
        Specify a training loss for this model
        """
        raise NotImplementedError

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
