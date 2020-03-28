import tensorflow as tf

from ...inducing_variables import (
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ...kernels import SeparateIndependent, LinearCoregionalization
from ..dispatch import conditional, sample_conditional
from ..util import sample_mvn, mix_latent_gp


@sample_conditional.register(
    object, SharedIndependentInducingVariables, LinearCoregionalization, object
)
def _sample_conditional(
    Xnew,
    inducing_variable,
    kernel,
    f,
    *,
    full_cov=False,
    full_output_cov=False,
    q_sqrt=None,
    white=False,
    num_samples=None,
):
    """
    `sample_conditional` will return a sample from the conditinoal distribution.
    In most cases this means calculating the conditional mean m and variance v and then
    returning m + sqrt(v) * eps, with eps ~ N(0, 1).
    However, for some combinations of Mok and Mof more efficient sampling routines exists.
    The dispatcher will make sure that we use the most efficent one.
    :return: [N, P] (full_output_cov = False) or [N, P, P] (full_output_cov = True)
    """
    if full_cov:
        raise NotImplementedError("full_cov not yet implemented")
    if full_output_cov:
        raise NotImplementedError("full_output_cov not yet implemented")

    ind_conditional = conditional.dispatch(
        object, SeparateIndependentInducingVariables, SeparateIndependent, object
    )
    g_mu, g_var = ind_conditional(
        Xnew, inducing_variable, kernel, f, white=white, q_sqrt=q_sqrt
    )  # [..., N, L], [..., N, L]
    g_sample = sample_mvn(g_mu, g_var, "diag", num_samples=num_samples)  # [..., (S), N, L]
    f_mu, f_var = mix_latent_gp(kernel.W, g_mu, g_var, full_cov, full_output_cov)
    f_sample = tf.tensordot(g_sample, kernel.W, [[-1], [-1]])  # [..., N, P]
    return f_sample, f_mu, f_var
