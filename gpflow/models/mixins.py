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
        Training loss for models that encapsulate their data.
        """
        return -self.maximum_a_posteriori_objective()

    def training_loss_closure(self, jit=True) -> Callable[[], tf.Tensor]:
        if jit:
            return tf.function(self.training_loss)
        return self.training_loss


class ExternalDataTrainingLossMixin:
    def training_loss(self, data):
        """
        Training loss for models that do not encapsulate the data.
        
        :param data: the data to be used for computing the model objective.
        """
        return -self.maximum_a_posteriori_objective(data)

    def training_loss_closure(
        self, data: Union[Data, collections.abc.Iterator], jit=True
    ) -> Callable[[], tf.Tensor]:
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
    def training_loss(self):
        """
        Training loss for gradient-based relaxation of MCMC models.
        """
        return -self.log_posterior_density()
