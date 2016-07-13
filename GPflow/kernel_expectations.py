import tensorflow as tf
import numpy as np
from . import kernels


def build_psi_stats(Z, kern, mu, S):
    if isinstance(kern, kernels.RBF):
        return build_psi_stats_rbf(Z, kern, mu, S)
    elif isinstance(kern, kernels.Linear):
        return build_psi_stats_linear(Z, kern, mu, S)
    elif isinstance(kern, kernels.Add):
        lkern = np.sort(kern.kern_list)  # so one order
        if len(lkern) == 2 and isinstance(lkern[0], kernels.Linear) and isinstance(lkern[1], kernels.RBF):
            # for RBF + Linear, we have an analytic solution:
            return build_psi_stats_rbf_plus_linear(Z, kern, mu, S)

        elif on_separate_dimensions(lkern):
            # for kernels additive over separate dimensions, the result is just the sum over each kernel.
            NxM = tf.pack([tf.shape(mu)[0], tf.shape(Z)[0]])
            MxM = tf.pack([tf.shape(Z)[0], tf.shape(Z)[0]])
            psi0, psi1, psi2 = tf.zeros((1,), tf.float64), tf.zeros(NxM, tf.float64), tf.zeros(MxM, tf.float64)
            for k in lkern:
                psi0_i, psi1_i, psi2_i = build_psi_stats(Z, k, mu, S)
                psi0 += psi0_i
                psi1 += psi1_i
                psi2 += psi2_i
            return psi0, psi1, psi2
        else:
            raise NotImplementedError("Psi-statistics: cannot add arbitrary kernels on overlapping dimensions")
    elif isinstance(kern, kernels.Prod):
        if on_separate_dimensions(lkern):
            # for kernels additive over separate dimensions, the result is just the product over each kernel.
            NxM = tf.pack([tf.shape(mu)[0], tf.shape(Z)[0]])
            MxM = tf.pack([tf.shape(Z)[0], tf.shape(Z)[0]])
            psi0, psi1, psi2 = tf.zeros((1,), tf.float64), tf.zeros(NxM, tf.float64), tf.zeros(MxM, tf.float64)
            for k in lkern:
                psi0_i, psi1_i, psi2_i = build_psi_stats(Z, k, mu, S)
                psi0 *= psi0_i
                psi1 *= psi1_i
                psi2 *= psi2_i
            return psi0, psi1, psi2
        else:
            raise NotImplementedError("Psi-statistics: cannot multiply arbitrary kernels on overlapping dimensions")
    elif is_one_dimensional(kern):
        return one_dimensional_psi_stats(Z, kern, mu, S)
    else:
        raise NotImplementedError("cannot compute Psi-statistics for this kernel")


def on_separate_dimensions(kernlist):
    """
    Take a list of kernels and return true if they operate on non-overlapping dimensions.
    """
    # TODO: what if active_dims is a slice? raise Exception
    sess = tf.InteractiveSession()
    dimlist = [sess.run(k.active_dims) for k in kernlist]
    overlapping = False
    for i, dims_i in enumerate(dimlist):
        for dims_j in dimlist[i+1:]:
            if np.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                overlapping = True
    return not overlapping


def is_one_dimensional(kern):
    sess = tf.InteractiveSession()
    dims = sess.run(kern.active_dims)
    return len(dims) == 1


def pad_inputs(kern, X):
    """
    prepend extra columns to X that will be sliced away by the kernel
    """
    return tf.concat(1, [tf.zeros(tf.pack([tf.shape(X)[0], kern.active_dims[0]]), tf.float64), X])


def one_dimensional_psi_stats(Z, kern, mu, S, numpoints=5):
    """
    This function computes the psi-statistics for an arbitrary kernel with only one input dimension.
    """
    # only use the active dimensions.
    mu, S = kern._slice(mu, S)
    Z, _ = kern._slice(Z, None)

    # compute a grid over which to compute approximate the integral
    gh_x, gh_w = np.polynomial.hermite.hermgauss(numpoints)
    gh_w /= np.sqrt(np.pi)
    X = gh_x * tf.sqrt(2.0 * S) + mu

    psi0 = reduce(tf.add, [tf.reduce_sum(kern.Kdiag(pad_inputs(kern, X[:, i:i+1])))*wi for i, wi in enumerate(gh_w)])

    # psi1
    KXZ = [kern.K(pad_inputs(kern, X[:, i:i+1]), Z) for i in range(numpoints)]
    psi1 = reduce(tf.add, [KXZ_i*wi for KXZ_i, wi in zip(KXZ, gh_w)])

    # psi2
    psi2 = reduce(tf.add, [tf.matmul(tf.transpose(KXZ_i), KXZ_i)*wi for KXZ_i, wi in zip(KXZ, gh_w)])

    return psi0, psi1, psi2


def build_psi_stats_linear(Z, kern, mu, S):
    # use only active dimensions
    mu, S = kern._slice(mu, S)  # only use the active dimensions.
    Z, _ = kern._slice(Z, None)

    psi0 = tf.reduce_sum(kern.variance*(tf.square(mu)+S))
    Zv = kern.variance * Z
    psi1 = tf.matmul(mu, tf.transpose(Zv))
    psi2 = tf.matmul(tf.reduce_sum(S, 0) * Zv, tf.transpose(Zv)) + tf.matmul(tf.transpose(psi1), psi1)
    return psi0, psi1, psi2


def build_psi_stats_rbf(Z, kern, mu, S):

    # use only active dimensions
    mu, S = kern._slice(mu, S)  # only use the active dimensions.
    Z, _ = kern._slice(Z, None)

    # psi0
    N = tf.shape(mu)[0]
    psi0 = tf.cast(N, tf.float64) * kern.variance

    # psi1
    lengthscale2 = tf.square(kern.lengthscales)
    psi1_logdenom = tf.expand_dims(tf.reduce_sum(tf.log(S / lengthscale2 + 1.), 1), 1)  # N x 1
    d = tf.square(tf.expand_dims(mu, 1)-tf.expand_dims(Z, 0))  # N x M x Q
    psi1_log = - 0.5 * (psi1_logdenom + tf.reduce_sum(d/tf.expand_dims(S+lengthscale2, 1), 2))
    psi1 = kern.variance * tf.exp(psi1_log)

    # psi2
    psi2_logdenom = -0.5 * tf.expand_dims(tf.reduce_sum(tf.log(2.*S/lengthscale2 + 1.), 1), 1)  # N # 1
    psi2_logdenom = tf.expand_dims(psi2_logdenom, 1)
    psi2_exp1 = 0.25 * tf.reduce_sum(tf.square(tf.expand_dims(Z, 1)-tf.expand_dims(Z, 0))/lengthscale2, 2)  # M x M
    psi2_exp1 = tf.expand_dims(psi2_exp1, 0)

    Z_hat = 0.5 * (tf.expand_dims(Z, 1) + tf.expand_dims(Z, 0))  # MxMxQ
    denom = 1./(2.*S+lengthscale2)
    a = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.square(mu)*denom, 1), 1), 1)  # N x 1 x 1
    b = tf.reduce_sum(tf.expand_dims(tf.expand_dims(denom, 1), 1) * tf.square(Z_hat), 3)  # N M M
    c = -2*tf.reduce_sum(tf.expand_dims(tf.expand_dims(mu*denom, 1), 1) * Z_hat, 3)  # N M M
    psi2_exp2 = a + b + c

    psi2 = tf.square(kern.variance) * tf.reduce_sum(tf.exp(psi2_logdenom - psi2_exp1 - psi2_exp2), 0)
    return psi0, psi1, psi2


def build_psi_stats_rbf_plus_linear(Z, kern, mu, S):
    # TODO: make sure the acvite dimensions are overlapping completely

    # use only active dimensions
    mu, S = kern._slice(mu, S)  # only use the active dimensions.
    Z, _ = kern._slice(Z, None)

    psi0_lin, psi1_lin, psi2_lin = build_psi_stats_linear(Z, kern.linear, mu, S)
    psi0_rbf, psi1_rbf, psi2_rbf = build_psi_stats_rbf(Z, kern.rbf, mu, S)
    psi0, psi1, psi2 = psi0_lin + psi0_rbf, psi1_lin + psi1_rbf, psi2_lin + psi2_rbf

    # extra terms for the 'interaction' of linear and rbf
    l2 = tf.square(kern.rbf.lengthscales)
    A = tf.expand_dims(1./S + 1./l2, 1)  # N x 1 x Q
    m = (tf.expand_dims(mu/S, 1) + tf.expand_dims(Z/l2, 0)) / A  # N x M x Q
    mTAZ = tf.reduce_sum(tf.expand_dims(m * kern.linear.variance, 1) *
                         tf.expand_dims(tf.expand_dims(Z, 0), 0), 3)  # N x M x M
    Z2 = tf.reduce_sum(tf.square(Z) / l2, 1)  # M,
    mu2 = tf.reduce_sum(tf.square(mu) / S, 1)  # N
    mAm = tf.reduce_sum(tf.square(m) * A, 2)  # N x M
    exp_term = tf.exp(-(tf.reshape(Z2, (1, -1)) + tf.reshape(mu2, (-1, 1))-mAm) / 2.)  # N x M
    psi2_extra = tf.reduce_sum(kern.rbf.variance *
                               tf.expand_dims(exp_term, 2) *
                               tf.expand_dims(tf.expand_dims(tf.reduce_prod(S, 1), 1), 2) *
                               tf.expand_dims(tf.reduce_prod(A, 2), 1) *
                               mTAZ, 0)

    psi2 = psi2 + psi2_extra + tf.transpose(psi2_extra)
    return psi0, psi1, psi2
