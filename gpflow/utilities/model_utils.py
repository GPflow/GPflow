import tensorflow as tf

from ..base import Parameter


def add_noise_cov(K: tf.Tensor, likelihood_variance: Parameter) -> tf.Tensor:
    """
    Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
    and I is the corresponding identity matrix.
    """
    k_diag = tf.linalg.diag_part(K)
    s_diag = tf.fill(tf.shape(k_diag), likelihood_variance)
    return tf.linalg.set_diag(K, k_diag + s_diag)


def add_linear_noise_cov(K: tf.Tensor, noise_variance: tf.Tensor) -> tf.Tensor:
    """
    Returns K + diag(σ²), where σ² is the likelihood noise variance (vector).
    """
    k_diag = tf.linalg.diag_part(K)
    return tf.linalg.set_diag(K, k_diag + tf.reshape(noise_variance, [-1]))

