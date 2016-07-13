import tensorflow as tf
import numpy as np
import GPflow


def on_separate_dimensions(kernlist):
    dimlist = [k.active_dims

def build_psi_stats(Z, kern, mu, S):
    if isinstance(kern, GPflow.kernels.RBF):
        return build_psi_stats_rbf(Z, kern, mu, S)
    elif isinstance(kern, GPflow.kernels.Linear):
        return build_psi_stats_linear(Z, kern, mu, S)
    elif isinstance(kern, GPflow.kernels.Add) and len(kern.kern_list) == 2:
        lkern = np.sort(kern.kern_list)  # so one order
        if len(lkern) == 2 and isinstance(lkern[0], GPflow.kernels.Linear) and isinstance(lkern[1], GPflow.kernels.RBF):
            return build_psi_stats_rbf_plus_linear(Z, kern, mu, S)
        elif on_separate_dimension(lkern)
            return ??
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def build_psi_stats_linear(Z, kern, mu, S):
    psi0 = tf.reduce_sum(kern.variance*(tf.square(mu)+S))
    Zv = kern.variance * Z
    psi1 = tf.matmul(mu, tf.transpose(Zv))
    psi2 = tf.matmul(tf.reduce_sum(S, 0) * Zv, tf.transpose(Zv)) + tf.matmul(tf.transpose(psi1), psi1)
    return psi0, psi1, psi2


def build_psi_stats_rbf(Z, kern, mu, S):
    N = tf.shape(mu)[0]

    # psi0
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
    psi0_lin, psi1_lin, psi2_lin = build_psi_stats_linear(Z, kern.linear, mu, S)
    psi0_rbf, psi1_rbf, psi2_rbf = build_psi_stats_rbf(Z, kern.rbf, mu, S)
    psi0, psi1, psi2 = psi0_lin + psi0_rbf, psi1_lin + psi1_rbf, psi2_lin + psi2_rbf

    # extra terms for the 'interaction' of linear and rbf
    l2 = tf.square(kern.rbf.lengthscales)
    A = tf.expand_dims(1./S + 1./l2, 1)  # N x 1 x Q
    m = (tf.expand_dims(mu/S, 1) + tf.expand_dims(Z/l2, 0)) / A  # N x M x Q
    mTAZ = tf.reduce_sum(tf.expand_dims(m * kern.linear.variance, 1)*tf.expand_dims(tf.expand_dims(Z, 0), 0), 3)  # N x M x M
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


# 
#     #psi2_extra = kern.linear.variance * 
# 
# 
# 
# 
# if __name__=='__main__':
#     import GPy
# 
#     Q = 3
#     N = 4
#     M = 2
# 
#     mu_np = np.random.randn(N, Q)
#     S_np = np.random.rand(N, Q)
#     Z_np = np.random.randn(M, Q)
# 
#     k_gpy = GPy.kern.RBF(Q)
# 
# 
#     
#     #build tf psi_stats
#     mu = tf.placeholder(tf.float64, [N, Q])
#     S = tf.placeholder(tf.float64, [N, Q])
#     Z = tf.placeholder(tf.float64, [M, Q])
#     x_free = tf.placeholder(tf.float64)
#     k = GPflow.kernels.RBF(Q, ARD=True)
# 
# 
#     k.make_tf_array(x_free)
#     with k.tf_mode():
#         tmp = build_psi_stats(Z, k, mu, S)
# 
#     p0_tf, p1_tf, p2_tf =  tf.Session().run(tmp, feed_dict={mu:mu_np, S:S_np, Z:Z_np, x_free:k.get_free_state()})
# 
#     p1_np = GPy.kern._src.psi_comp.rbf_psi_comp.__psi1computations(1, 1, Z_np, mu_np, S_np)
#     p2_np = GPy.kern._src.psi_comp.rbf_psi_comp.__psi2computations(1, 1, Z_np, mu_np, S_np).sum(0)
