import tensorflow as tf

from .. import mean_functions
from .. import covariances
from ..expectations import expectation
from ..inducing_variables import InducingVariables, InducingPoints
from ..kernels import Kernel
from ..probability_distributions import Gaussian
from ..config import default_float, default_jitter


def uncertain_conditional(
    Xnew_mu: tf.Tensor,
    Xnew_var: tf.Tensor,
    inducing_variable: InducingVariables,
    kernel: Kernel,
    q_mu,
    q_sqrt,
    *,
    mean_function=None,
    full_output_cov=False,
    full_cov=False,
    white=False,
):
    """
    Calculates the conditional for uncertain inputs Xnew, p(Xnew) = N(Xnew_mu, Xnew_var).
    See ``conditional`` documentation for further reference.
    :param Xnew_mu: mean of the inputs, size [N, D]in
    :param Xnew_var: covariance matrix of the inputs, size [N, n, n]
    :param inducing_variable: gpflow.InducingVariable object, only InducingPoints is supported
    :param kernel: gpflow kernel object.
    :param q_mu: mean inducing points, size [M, Dout]
    :param q_sqrt: cholesky of the covariance matrix of the inducing points, size [t, M, M]
    :param full_output_cov: boolean wheter to compute covariance between output dimension.
                            Influences the shape of return value ``fvar``. Default is False
    :param white: boolean whether to use whitened representation. Default is False.
    :return fmean, fvar: mean and covariance of the conditional, size ``fmean`` is [N, Dout],
            size ``fvar`` depends on ``full_output_cov``: if True ``f_var`` is [N, t, t],
            if False then ``f_var`` is [N, Dout]
    """

    if not isinstance(inducing_variable, InducingPoints):
        raise NotImplementedError

    if full_cov:
        # TODO(VD): ``full_cov`` True would return a ``fvar`` of shape [N, N, D, D],
        # encoding the covariance between input datapoints as well.
        # This is not implemented as this inducing_variable is only used for plotting purposes.
        raise NotImplementedError

    pXnew = Gaussian(Xnew_mu, Xnew_var)

    num_data = tf.shape(Xnew_mu)[0]  # number of new inputs (N)
    num_ind, num_func = tf.unstack(
        tf.shape(q_mu), num=2, axis=0
    )  # number of inducing points (M), output dimension (D)
    q_sqrt_r = tf.linalg.band_part(q_sqrt, -1, 0)  # [D, M, M]

    eKuf = tf.transpose(expectation(pXnew, (kernel, inducing_variable)))  # [M, N] (psi1)
    Kuu = covariances.Kuu(inducing_variable, kernel, jitter=default_jitter())  # [M, M]
    Luu = tf.linalg.cholesky(Kuu)  # [M, M]

    if not white:
        q_mu = tf.linalg.triangular_solve(Luu, q_mu, lower=True)
        Luu_tiled = tf.tile(
            Luu[None, :, :], [num_func, 1, 1]
        )  # remove line once issue 216 is fixed
        q_sqrt_r = tf.linalg.triangular_solve(Luu_tiled, q_sqrt_r, lower=True)

    Li_eKuf = tf.linalg.triangular_solve(Luu, eKuf, lower=True)  # [M, N]
    fmean = tf.linalg.matmul(Li_eKuf, q_mu, transpose_a=True)

    eKff = expectation(pXnew, kernel)  # N (psi0)
    eKuffu = expectation(
        pXnew, (kernel, inducing_variable), (kernel, inducing_variable)
    )  # [N, M, M] (psi2)
    Luu_tiled = tf.tile(
        Luu[None, :, :], [num_data, 1, 1]
    )  # remove this line, once issue 216 is fixed
    Li_eKuffu = tf.linalg.triangular_solve(Luu_tiled, eKuffu, lower=True)
    Li_eKuffu_Lit = tf.linalg.triangular_solve(
        Luu_tiled, tf.linalg.adjoint(Li_eKuffu), lower=True
    )  # [N, M, M]
    cov = tf.linalg.matmul(q_sqrt_r, q_sqrt_r, transpose_b=True)  # [D, M, M]

    if mean_function is None or isinstance(mean_function, mean_functions.Zero):
        e_related_to_mean = tf.zeros((num_data, num_func, num_func), dtype=default_float())
    else:
        # Update mean: \mu(x) + m(x)
        fmean = fmean + expectation(pXnew, mean_function)

        # Calculate: m(x) m(x)^T + m(x) \mu(x)^T + \mu(x) m(x)^T,
        # where m(x) is the mean_function and \mu(x) is fmean
        e_mean_mean = expectation(pXnew, mean_function, mean_function)  # [N, D, D]
        Lit_q_mu = tf.linalg.triangular_solve(Luu, q_mu, adjoint=True)
        e_mean_Kuf = expectation(pXnew, mean_function, (kernel, inducing_variable))  # [N, D, M]
        # einsum isn't able to infer the rank of e_mean_Kuf, hence we explicitly set the rank of the tensor:
        e_mean_Kuf = tf.reshape(e_mean_Kuf, [num_data, num_func, num_ind])
        e_fmean_mean = tf.einsum("nqm,mz->nqz", e_mean_Kuf, Lit_q_mu)  # [N, D, D]
        e_related_to_mean = e_fmean_mean + tf.linalg.adjoint(e_fmean_mean) + e_mean_mean

    if full_output_cov:
        fvar = (
            tf.linalg.diag(tf.tile((eKff - tf.linalg.trace(Li_eKuffu_Lit))[:, None], [1, num_func]))
            + tf.linalg.diag(tf.einsum("nij,dji->nd", Li_eKuffu_Lit, cov))
            +
            # tf.linalg.diag(tf.linalg.trace(tf.linalg.matmul(Li_eKuffu_Lit, cov))) +
            tf.einsum("ig,nij,jh->ngh", q_mu, Li_eKuffu_Lit, q_mu)
            -
            # tf.linalg.matmul(q_mu, tf.linalg.matmul(Li_eKuffu_Lit, q_mu), transpose_a=True) -
            fmean[:, :, None] * fmean[:, None, :]
            + e_related_to_mean
        )
    else:
        fvar = (
            (eKff - tf.linalg.trace(Li_eKuffu_Lit))[:, None]
            + tf.einsum("nij,dji->nd", Li_eKuffu_Lit, cov)
            + tf.einsum("ig,nij,jg->ng", q_mu, Li_eKuffu_Lit, q_mu)
            - fmean ** 2
            + tf.linalg.diag_part(e_related_to_mean)
        )

    return fmean, fvar
