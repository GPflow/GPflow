from typing import Any, Callable

import tensorflow as tf

from ..base import TensorType
from ..experimental.check_shapes import check_shapes


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
    "likelihood_variance: []",
    "return: [batch..., N, N]",
)
def add_noise_cov(K: tf.Tensor, likelihood_variance: TensorType) -> tf.Tensor:
    """
    Returns K + σ² I, where σ² is the likelihood noise variance and I is the corresponding identity
    matrix.
    """
    k_diag = tf.linalg.diag_part(K)
    s_diag = tf.fill(tf.shape(k_diag), likelihood_variance)
    return tf.linalg.set_diag(K, k_diag + s_diag)
