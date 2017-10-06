from matplotlib import pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
from tensorflow.python import debug as tf_debug


import sys
import csv

import numpy as np


import gpflow

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
    p = m.predict_f(Xplot)[0]
    ax.plot(Xtrain[:, 0][Ytrain[:, 0] == 1], Xtrain[:, 1][Ytrain[:, 0] == 1], 'o', color=col1,
            mew=0, alpha=0.5)
    ax.plot(Xtrain[:, 0][Ytrain[:, 0] == 0], Xtrain[:, 1][Ytrain[:, 0] == 0], 'o', color=col2,
            mew=0, alpha=0.5)
    if hasattr(m, 'Z'):
        ax.plot(m.Z.value[:, 0], m.Z.value[:, 1], 'ko', mew=0, ms=4)

    if np.min(p) < -0.1:
        lin = np.linspace(-1., 1., 10)
    else:
        lin = np.linspace(0., 1., 10)
    ax.contour(xx, yy, p.reshape(*xx.shape), lin, colors='k', linewidths=1.8, zorder=100)


def main():
    f, axarr = plt.subplots(1, 2)
    m = gpflow.models.VGP(Xtrain, Ytrain,
                                kern=gpflow.kernels.RBF(2),
                                likelihood=gpflow.likelihoods.Bernoulli())
    m.kern.lengthscales.set_trainable(False)
    m.kern.variance.set_trainable(False)
    m.compile()
    gpflow.train.ScipyOptimizer(options=dict(maxiter=20)).minimize(m)
    plot(m, axarr[0])
    # print(m.kern.lengthscales.read_value())
    # #
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    m2 = gpflow.models.EPBinClassGP(Xtrain, Ytrain, kern=gpflow.kernels.RBF(2))
    #m2 = gpflow.models.EPBinClassGP(Xtrain[:10, :], Ytrain[:10, :], kern=gpflow.kernels.RBF(2))
    #m2 = gpflow.models.EPBinClassGP(Xtrain, Ytrain, kern=gpflow.kernels.RBF(2))

    #m2.kern.lengthscales = 0.6930140075902973
    #m2.kern.variance = 4.248435519556653

    #[print(a.read_value()) for a in [m2.kern.lengthscales, m2.kern.variance]]

    m2.compile(session=sess)
    _, _, num_iter = m2.run_ep()
    print(num_iter)
    #gpflow.train.ScipyOptimizer(options=dict(maxiter=20)).minimize(m2)
    #m2.run_ep()
    plot(m2, axarr[1])

    plt.show()


if __name__ == '__main__':
    main()
