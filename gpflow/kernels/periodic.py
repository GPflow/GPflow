from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from ..base import Parameter
from ..utilities import positive
from .base import Kernel
from .stationaries import Stationary


class Periodic(Kernel):
    """
    The periodic family of kernels. Can be used to wrap any Stationary kernel
    to transform it into a periodic version. The canonical form (based on the
    SquaredExponential kernel) can be found in Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    The derivation can be achieved by mapping the original inputs through the
    transformation u = (cos(x), sin(x)).

    For the SquaredExponential base kernel, the result can be expressed as:

        k(r) =  σ² exp{ -0.5 sin²(π r / γ) / ℓ²}

    where:
    r is the Euclidean distance between the input points
    ℓ is the lengthscale parameter,
    σ² is the variance parameter,
    γ is the period parameter.

    (note that usually we have a factor of 4 instead of 0.5 in front but this
    is absorbed into the lengthscale hyperparameter).
    """

    def __init__(self, base: Stationary, period: Union[float, List[float]] = 1.0,
                 **kwargs):
        """
        :param base: the base kernel to make periodic; must inherit from Stationary
            Note that active_dims should be specified in the base kernel.
        :param period: the period; to induce a different period per active dimension
            this must be initialized with an array the same length as the number
            of active dimensions e.g. [1., 1., 1.]
        """
        if not isinstance(base, Stationary):
            raise TypeError("Periodic requires a Stationary kernel as the `base`")

        if 'active_dims' in kwargs:
            if self._normalize_active_dims(kwargs['active_dims']) != base.active_dims:
                raise ValueError("active_dims must be consistent with base.active_dims")

        super().__init__()
        self.base = base
        self.period = Parameter(period, transform=positive())
        self.base._validate_ard_active_dims(self.period)

    @property
    def active_dims(self):
        return self.base.active_dims

    @active_dims.setter
    def active_dims(self, value):
        self.base.active_dims = value

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        return self.base.K_diag(X)

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        # TODO: does not support kernel broadcasting
        f = tf.expand_dims(X, 1)  # now [N, 1, D]
        f2 = tf.expand_dims(X2, 0)  # now [1, M, D]

        r = np.pi * (f - f2) / self.period
        scaled_sine = tf.sin(r) / self.base.lengthscale
        if hasattr(self.base, "K_r"):
            sine_r = tf.reduce_sum(tf.abs(scaled_sine), -1)
            K = self.base.K_r(sine_r)
        else:
            sine_r2 = tf.reduce_sum(tf.square(scaled_sine), -1)
            K = self.base.K_r2(sine_r2)
        return K
