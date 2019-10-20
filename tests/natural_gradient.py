import tensorflow as tf

import gpflow
from gpflow.config import default_float
from gpflow.optimizers import NaturalGradient
from gpflow.utilities import set_trainable


def vgp_vs_gpr():
    """
    With a Gaussian likelihood the Gaussian variational (VGP) model should be equivalent to the exact
    regression model (GPR) after a single nat grad step of size 1
    """
    N, D = 3, 2
    x = tf.random.normal((N, D), dtype=default_float())
    y = tf.random.normal((N, 1), dtype=default_float())
    data = (x, y)
    kernel = gpflow.kernels.RBF()
    likelihood = gpflow.likelihoods.Gaussian()
    likelihood_variance = 0.1
    likelihood.variance.assign(likelihood_variance)

    vgp = gpflow.models.VGP(data, kernel, likelihood)
    gpr = gpflow.models.GPR(data, kernel)
    gpr.likelihood.variance.assign(likelihood_variance)

    set_trainable(vgp, False)
    vgp.q_mu.trainable = True
    vgp.q_sqrt.trainable = True

    def loss_cb() -> tf.Tensor:
        return vgp.neg_log_marginal_likelihood()

    vgp_ll_before = vgp.log_likelihood()
    gpr_ll_before = gpr.log_likelihood()

    assert vgp_ll_before != gpr_ll_before

    params = [(vgp.q_mu, vgp.q_sqrt)]
    opt = NaturalGradient(1.)
    opt.minimize(loss_cb, vars_list=params)

    vgp_ll_after = vgp.log_likelihood()
    gpr_ll_after = gpr.log_likelihood()

    print("finished")

    # assert_allclose(m_gpr.compute_log_likelihood(), m_vgp.compute_log_likelihood(), atol=1e-4)


vgp_vs_gpr()