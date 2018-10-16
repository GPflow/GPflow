from typing import Callable

import tensorflow as tf

from ..features import InducingFeature
from ..kernels import Kernel
from ..util import create_logger
from .dispatch import conditional, sample_conditional
from .util import sample_mvn

logger = create_logger()


@sample_conditional.register(object, InducingFeature, Kernel, object)
def _sample_conditional(Xnew: tf.Tensor,
                        feature: InducingFeature,
                        kernel: Kernel,
                        function: Callable, *,
                        full_output_cov=False, q_sqrt=None, white=False):
    """
    `sample_conditional` will return a sample from the conditinoal distribution.
    In most cases this means calculating the conditional mean m and variance v and then
    returning m + sqrt(v) * eps, with eps ~ 𝒩(0, 1).
    However, for some combinations of Mok and Mof more efficient sampling routines exists.
    The dispatcher will make sure that we use the most efficent one.

    :return: [N, P] (full_output_cov = False) or [N, P, P] (full_output_cov = True)
    """
    logger.debug("Sample conditional: InducingFeature Kernel")
    mean, var = conditional(Xnew, feature, kernel, function,
                            full_cov=False, full_output_cov=full_output_cov,
                            q_sqrt=q_sqrt, white=white)  # N x P, N x P (x P)
    cov_structure = "full" if full_output_cov else "diag"
    return sample_mvn(mean, var, cov_structure)


@sample_conditional.register(object, object, Kernel, object)
def _sample_conditional(Xnew: tf.Tensor, X: tf.Tensor, kernel: Kernel, function: Callable, *, q_sqrt=None, white=False):
    logger.debug("Sample conditional: Kernel")
    mean, var = conditional(Xnew, X, kernel, function, q_sqrt=q_sqrt, white=white, full_cov=False)  # [N, P], [N, P]
    return sample_mvn(mean, var, "diag")  # [N, P]
