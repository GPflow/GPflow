import tensorflow as tf

from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from .dispatch import conditional, sample_conditional
from .util import sample_mvn


@sample_conditional.register(object, object, Kernel, object)
@sample_conditional.register(object, InducingVariables, Kernel, object)
def _sample_conditional(
    Xnew: tf.Tensor,
    inducing_variable: InducingVariables,
    kernel: Kernel,
    f: tf.Tensor,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
    num_samples=None,
):
    """
    `sample_conditional` will return a sample from the conditional distribution.
    In most cases this means calculating the conditional mean m and variance v and then
    returning m + sqrt(v) * eps, with eps ~ N(0, 1).
    However, for some combinations of Mok and Mof more efficient sampling routines exists.
    The dispatcher will make sure that we use the most efficient one.
    :return: samples, mean, cov
        samples has shape [num_samples, N, P] or [N, P] if num_samples is None
        mean and cov as for conditional()
    """

    if full_cov and full_output_cov:
        msg = "The combination of both `full_cov` and `full_output_cov` is not permitted."
        raise NotImplementedError(msg)

    mean, cov = conditional(
        Xnew,
        inducing_variable,
        kernel,
        f,
        q_sqrt=q_sqrt,
        white=white,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
    )
    if full_cov:
        # mean: [..., N, P]
        # cov: [..., P, N, N]
        mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
        samples = sample_mvn(
            mean_for_sample, cov, "full", num_samples=num_samples
        )  # [..., (S), P, N]
        samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
    else:
        # mean: [..., N, P]
        # cov: [..., N, P] or [..., N, P, P]
        cov_structure = "full" if full_output_cov else "diag"
        samples = sample_mvn(mean, cov, cov_structure, num_samples=num_samples)  # [..., (S), N, P]

    return samples, mean, cov
