""" Simplified initalisation code adapted from Trieste """

import tensorflow as tf

from gpflow import Module
from gpflow.utilities import read_values, multiple_assign, print_summary


def randomize_hyperparameters(object: Module) -> None:
    """
    Sets hyperparameters to random samples from their constrained domains or (if not constraints
    are available) their prior distributions.
    :param object: Any gpflow Module.
    """
    for param in object.trainable_parameters:
        if param.prior is not None:
            sample_shape = param.shape
            param.assign(param.prior.sample(sample_shape))

def find_best_model_initialization(model, n_inits: int):
        @tf.function
        def evaluate_loss_of_random_model_parameters() -> tf.Tensor:
                randomize_hyperparameters(model)
                return model.training_loss()

        current_best_parameters = read_values(model)
        min_loss = model.training_loss()
        print("Starting initialisation")

        for i in range(n_inits):
                try:
                        train_loss = evaluate_loss_of_random_model_parameters()
                except tf.errors.InvalidArgumentError:  # allow badly specified kernel params
                        train_loss = 1e100

                if train_loss < min_loss:  # only keep best kernel params
                        min_loss = train_loss
                        current_best_parameters = read_values(model)
                        tf.print("New max likelihood found at iter ", i)

        multiple_assign(model, current_best_parameters)
        print("Finished initialisation:")
        print_summary(model)

        return model

