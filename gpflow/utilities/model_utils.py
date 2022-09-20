from typing import Any, Callable

import tensorflow as tf

from ..base import TensorType
from ..experimental.check_shapes import check_shapes
from ..likelihoods import Gaussian


def assert_params_false(
    called_method: Callable[..., Any],
    **kwargs: bool,
) -> None:
    """
    Asserts that parameters are ``False``.

    :param called_method: The method or function that is calling this. Used for nice error messages.
    :param kwargs: Parameters that must be ``False``.
    :raises NotImplementedError: If any ``kwargs`` are ``True``.
    """
    errors_str = ", ".join(f"{param}={value}" for param, value in kwargs.items() if value)
    if errors_str:
        raise NotImplementedError(
            f"{called_method.__qualname__} does not currently support: {errors_str}"
        )


@check_shapes(
    "K: [batch..., N, N]",
    "likelihood_variance: [broadcast batch..., broadcast N]",
    "return: [batch..., N, N]",
)
def add_noise_cov(K: tf.Tensor, likelihood_variance: TensorType) -> tf.Tensor:
    """
    Returns K + σ², where σ² is the diagonal likelihood noise variance.
    """
    k_diag = tf.linalg.diag_part(K)
    return tf.linalg.set_diag(K, k_diag + likelihood_variance)


@check_shapes(
    "K: [batch..., N, N]",
    "X: [batch..., N, D]",
    "return: [batch..., N, N]",
)
def add_likelihood_noise_cov(K: tf.Tensor, likelihood: Gaussian, X: TensorType) -> tf.Tensor:
    """
    Returns K + σ², where σ² is the likelihood noise variance.
    """
    return add_noise_cov(K, tf.squeeze(likelihood.variance_at(X), axis=-1))
