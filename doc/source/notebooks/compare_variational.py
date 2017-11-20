import gpflow
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

nPoints = 3
rng =np.random.RandomState(4)
X = np.atleast_2d(rng.rand(nPoints)*10).T
Y = np.sin(X) + 0.9 * np.cos(X*1.6) + np.random.randn(*X.shape)* 0.8

def kernel():
    return gpflow.kernels.RBF(1)

plt.plot(X, Y, 'kx')

m2 = gpflow.models.VGP(X, Y, kern=kernel(), likelihood=gpflow.likelihoods.Gaussian())
m3 = gpflow.models.SVGP(X, Y, kern=kernel(), likelihood=gpflow.likelihoods.Gaussian(), Z=X.copy(), q_diag=False, whiten=True)
m4 = gpflow.models.SVGP(X, Y, kern=kernel(), likelihood=gpflow.likelihoods.Gaussian(), Z=X.copy(), q_diag=False, whiten=False)

m3.feature.trainable = False
m4.feature.trainable = False

model_list = [m2,m3,m4]

for m in model_list:
    m.kern.lengthscales.fixed = True
    m.kern.variance.fixed = True
    m.likelihood.variance.fixed = True

opt = gpflow.train.ScipyOptimizer()
opt.minimize(m2, maxiter=100000)
opt.minimize(m3, maxiter=100000)
opt.minimize(m4, maxiter=100000)


xx = np.linspace(-1, 11, 100)[:,None]

def plot(m, color='b'):
    mu, var = m.predict_y(xx)
    plt.plot(xx, mu, color, lw=2)
    plt.plot(xx, mu+ 2*np.sqrt(var), color, xx, mu-2*np.sqrt(var), color, lw=1)

plt.figure(figsize=(12,8))
plot(m2, 'r')
plot(m3, 'g')
plot(m4, 'y')
plt.plot(X, Y, 'kx')
embed()
