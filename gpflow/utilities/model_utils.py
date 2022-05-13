import tensorflow as tf

from ..base import TensorType
from ..experimental.check_shapes import check_shapes


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
