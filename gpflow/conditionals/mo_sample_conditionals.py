import tensorflow as tf

from ..features import MixedKernelSharedMof, SeparateIndependentMof
from ..kernels import SeparateIndependentMok, SeparateMixedMok
from ..util import create_logger
from .dispatch import conditional, sample_conditional
from .util import sample_mvn

logger = create_logger()


@sample_conditional.register(object, MixedKernelSharedMof, SeparateMixedMok, object)
def _sample_conditional(Xnew, feat, kern, f, *, full_output_cov=False, q_sqrt=None, white=False):
    """
    `sample_conditional` will return a sample from the conditinoal distribution.
    In most cases this means calculating the conditional mean m and variance v and then
    returning m + sqrt(v) * eps, with eps ~ N(0, 1).
    However, for some combinations of Mok and Mof more efficient sampling routines exists.
    The dispatcher will make sure that we use the most efficent one.

    :return: N x P (full_output_cov = False) or N x P x P (full_output_cov = True)
    """
    logger.debug("sample conditional: MixedKernelSharedMof, SeparateMixedMok")
    independent_cond = conditional.dispatch(object, SeparateIndependentMof, SeparateIndependentMok, object)
    g_mu, g_var = independent_cond(Xnew, feat, kern, f, white=white, q_sqrt=q_sqrt,
                                   full_output_cov=False, full_cov=False)  # N x L, N x L
    g_sample = sample_mvn(g_mu, g_var, "diag")  # N x L
    f_sample = tf.einsum("pl, nl -> np", kern.W(), g_sample)
    return f_sample
