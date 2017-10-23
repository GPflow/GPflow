from matplotlib import pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
from tensorflow.python import debug as tf_debug
# ^ import the above for the TF debugger.

import numpy as np


import gpflow
from gpflow_external_models import ep_approximated_like

Xtrain = np.loadtxt('data/banana_X_train', delimiter=',')
Ytrain = np.loadtxt('data/banana_Y_train', delimiter=',').reshape(-1,1)


def gridParams():
    mins = [-3.25, -2.85]
    maxs = [3.65, 3.4]
    nGrid = 50
    xspaced = np.linspace(mins[0], maxs[0], nGrid)
    yspaced = np.linspace(mins[1], maxs[1], nGrid)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T
    return mins, maxs, xx, yy, Xplot


def plot(m, ax):
    col1 = '#0172B2'
    col2 = '#CC6600'
    mins, maxs, xx, yy, Xplot = gridParams()
    p, _ = m.predict_y(Xplot)
    ax.plot(Xtrain[:, 0][Ytrain[:, 0] == 1], Xtrain[:, 1][Ytrain[:, 0] == 1], 'o', color=col1,
            mew=0, alpha=0.5)
    ax.plot(Xtrain[:, 0][Ytrain[:, 0] == 0], Xtrain[:, 1][Ytrain[:, 0] == 0], 'o', color=col2,
            mew=0, alpha=0.5)
    if hasattr(m, 'Z'):
        ax.plot(m.Z.value[:, 0], m.Z.value[:, 1], 'ko', mew=0, ms=4)

    if np.min(p) < -0.1:
        lin = np.linspace(-1., 1., 10)
    else:
        print("10 levels between 0 and 10")
        lin = np.linspace(0., 1., 10)
    ax.contour(xx, yy, p.reshape(*xx.shape), lin, colors='k', linewidths=1.8, zorder=100)
    ax.set_title("{}".format(type(m)))


def main():
    f, axarr = plt.subplots(1, 2, figsize=(15, 7.5))
    TRAIN_KERNEL = True
    USE_ADAM = False

    # VGP Model:
    print("Running VGP model.")
    m = gpflow.models.VGP(Xtrain, Ytrain,
                                kern=gpflow.kernels.RBF(2),
                                likelihood=gpflow.likelihoods.Bernoulli())
    if not TRAIN_KERNEL:
        m.kern.lengthscales.set_trainable(False)
        m.kern.variance.set_trainable(False)
    m.compile()
    print("VGP model's initial model log likelihood: {}".format(m.compute_log_likelihood()))
    if USE_ADAM:
        gpflow.train.AdamOptimizer().minimize(m, maxiter=500)
    else:
        gpflow.train.ScipyOptimizer(options=dict(maxiter=100)).minimize(m)
    plot(m, axarr[0])
    print("VGP model's final model log likelihood: {}".format(m.compute_log_likelihood()))
    print("VGP model's final kernel variance: {}".format(m.kern.variance.read_value()))
    print("VGP model's final kernel lengthscale: {}".format(m.kern.lengthscales.read_value()))
    print("=================================\n\n")

    # EP Binary Classification Model:
    print("Running Binary EP Classification Model")
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # ^ if debugging run with this uncommented and ` python -m ep_classification_demo --debug`
    m2 = ep_approximated_like.EPLikeApproxGP(Xtrain, Ytrain, kern=gpflow.kernels.RBF(2),
                                             likelihood=gpflow.likelihoods.Bernoulli(),
                                             use_cache_on_like=False)
    m2.compile(session=sess)
    tau_tilde, nu_tilde, num_iter = m2.run_ep()
    print("Number iterations pre optimisation for EP convergence {}".format(num_iter))
    print("EPBinGP model's initial model log likelihood: {}".format(m2.compute_log_likelihood()))
    if USE_ADAM:
        gpflow.train.AdamOptimizer().minimize(m2, maxiter=500)
    else:
        gpflow.train.ScipyOptimizer(options=dict(maxiter=100)).minimize(m2)
    tau_tilde, nu_tilde, num_iter = m2.run_ep()
    print("Number iterations post optimisation for EP convergence {}".format(num_iter))
    plot(m2, axarr[1])
    print("EPBinGP model's final model log likelihood: {}".format(m2.compute_log_likelihood()))
    print("EPBinGP model's final kernel variance: {}".format(m2.kern.variance.read_value()))
    print("EPBinGP model's final kernel lengthscale: {}".format(m2.kern.lengthscales.read_value()))
    print("=================================\n\n")
    plt.show()




if __name__ == '__main__':
    main()
