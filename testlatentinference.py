import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12,6)
matplotlib.style.use('ggplot')
import GPflow
import numpy as np
import tensorflow as tf
np.random.seed(0)

Q = 1 # latent dimension
D = 2 # output dimension
N = 100
lengthscale = 3.5
variance = 1.
noise_var = 0.001

rng = np.random.RandomState(5) # 5 happens to be a pretty draw.
X = rng.randn(N, Q)
k = GPflow.kernels.PeriodicKernel(Q, variance=variance, lengthscales=lengthscale, period=3, active_dims=[0])
K = k.compute_K(X, X)
Y = rng.multivariate_normal(np.zeros(N), K + np.eye(N) * noise_var, D).T

XPCA = GPflow.gplvm.PCA_reduce(Y, 1)
m = GPflow.gplvm.BayesianGPLVM(Y=Y, kern=k, X_mean=XPCA, X_var=np.ones((N, Q)) * 0.01, Z=np.random.randn(10, Q), M=10)
m.likelihood.variance = noise_var

def plot():
    f, axes = plt.subplots(1, 3, figsize=(16, 5))
    Xtest = np.linspace(m.X_mean.value.min(), m.X_mean.value.max(), 100)[:,None]
    mu, var = m.predict_y(Xtest)
    # samples = m.predict_f_samples(X, 5) suffers numerical problem
    _, covar = m.predict_f_full_cov(Xtest)
    samples = [np.random.multivariate_normal(mu_i, var_i, 5) for mu_i, var_i in zip(mu.T, covar.T)]

    for i, ax in enumerate(axes[:2]):
        #ax.plot(m.X_mean.value, m.Y[:,i], 'kx')
        ax.plot(Xtest, mu[:,i], 'b')
        ax.plot(Xtest, mu[:,i] + 2*np.sqrt(var[:,i]), 'b--')
        ax.plot(Xtest, mu[:,i] - 2*np.sqrt(var[:,i]), 'b--')
        ax.scatter(m.X_mean.value, m.Y.value[:,i], 100, X, lw=2, cmap=plt.cm.viridis)
        ax.plot(Xtest.flatten(), samples[i].T, 'b', lw=0.3)

        ax.set_xlabel('X (inferred)')
        ax.set_ylabel('Y_%i (data)'%i)


    #axes[2].plot(m.Y[:,0], m.Y[:,1], 'kx')
    axes[2].scatter(m.Y.value[:,0], m.Y.value[:,1], 100, X, lw=2, cmap=plt.cm.viridis)
    axes[2].plot(mu[:,0], mu[:,1], 'b')
    for s1, s2 in zip(*samples):
        axes[2].plot(s1, s2, 'b', lw=0.3)
    axes[2].set_xlabel('Y_0 (data)')
    axes[2].set_ylabel('Y_1 (data)')

m.optimize()
Xmu, Xvar = m.infer_latent_inputs(np.atleast_2d(Y[0,:]), disp=True)
print(m)
print(Xmu)
print(Xvar)