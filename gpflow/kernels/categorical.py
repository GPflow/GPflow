from typing import Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ..base import Parameter, TensorType
from .. import set_trainable, default_int
from . import Kernel
tfd = tfp.distributions
st = tfp.bijectors.Softplus().inverse

def latent_from_labels(Z: tf.Tensor, labels: TensorType) -> TensorType:
    """
    Converts a tensor of labels into a tensor of the latent space values of those labels.
    :param Z: Z is a tensor whose entries are the latent space values of the relevant label
        i.e. Z[3] is the value of label 3.
        [num_labels]
    :param X: X is the input data.  The last column should contain the labels as integers.
        [..., N, D+1]
    :return: [..., N, D + label_dim]
    """
    indices = tf.cast(labels, default_int())  # [..., N]
    return tf.gather(Z, indices)  # [..., N, label_dim]

def _concat_inputs_with_latents(Z: tf.Tensor, X: TensorType) -> TensorType:
    """
    Transforms a dataset X from category space into the latent space.
    :param Z: Z is a tensor whose entries are the latent space values of the relevant label
        i.e. Z[3] is the value of label 3.
        [num_labels]
    :param X: X is the input data.  The last column should contain the labels as integers.
        [..., N, D+1]
    :return: [..., N, D + label_dim]
    """
    labels = X[..., -1]
    latent_values = latent_from_labels(Z, labels)

    return tf.concat([X[..., :-1], latent_values], axis=-1)  # [..., N, x_dim + label_dim]

class CategoricalKernel(Kernel):
    """
    A class that implements a categorical latent space wrapper for other Kernels.  It works
     by dynamically replacing integer labels in the input data with values in a latent space (Z),
     wrapping two kernels which are combined multiplicatively and otherwise work as normal.
    Kernel 1 is any kernel of your choice which will deal with the non-categorical dimensions.
    Kernel 2 deals with the categorical dimension.  It is fixed in training, in order to reduce the number
     of degrees of freedom in optimisation.  You may also find fixing the lengthscale to be useful here.
     Note that Z is parameterised by the differences of latent space values for each category for the same reason.
    Multiple categories can be included by wrapping multiple layers of CategoricalKernel.
    :param wrapped_kernel_1: The non-categorical kernel.
    :param wrapped_kernel_2: The categorical kernel.  Set its active_dims to the label dimension.
    :param num_labels: The number of labels
    """
    def __init__(self, wrapped_kernel_1: Kernel, wrapped_kernel_2: Kernel, num_labels: int, *args, **kwargs):
        # wrapped_kernel_2 is the one that trains the categorical variables
        set_trainable(wrapped_kernel_2, False)
        self.wrapped_kernel = wrapped_kernel_1 * wrapped_kernel_2
        label_dim = 1
        self._Z_deltas = Parameter(
            np.random.random((num_labels-1, label_dim))*wrapped_kernel_2.lengthscales * 10)
        super().__init__(*args, **kwargs)
        
    def _concat_inputs_with_latents(self, X):
        return _concat_inputs_with_latents(self.Z, X)
    
    @property
    def Z(self) -> tf.Tensor:
        """
        Z is a tensor whose entries are the latent space values of the relevant label
        i.e. Z[3] is the value of label 3.
        """
        Z = tf.concat([tf.constant(0,shape=(1,),dtype=tf.float64), tf.squeeze(self._Z_deltas)], 0)
        m = tf.linalg.band_part(tf.ones([tf.size(Z), tf.size(Z)], dtype=tf.float64), -1, 0)
        return tf.expand_dims(tf.linalg.matvec(m, Z), -1)

    def K(self, X: TensorType, X2: Optional[TensorType]=None) -> tf.Tensor:
        return self.wrapped_kernel.K(
            self._concat_inputs_with_latents(X),
            self._concat_inputs_with_latents(X2) if X2 is not None else None
        )
    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self.wrapped_kernel.K_diag(self._concat_inputs_with_latents(X))