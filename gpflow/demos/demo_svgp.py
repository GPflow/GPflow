

import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

from gpflow.models.sparsegp import SparseVariationalGP

plt.style.use('ggplot')

xmin, xmax = 0, 5
N = 1000
X = np.linspace(xmin, xmax, N).reshape(-1, 1)
Y = np.cos(X * 4) + np.random.randn(N, 1)*.5
data = (X, Y)

plt.plot(X, Y, 'kx', mew=2);
plt.show()

# ## Building the model

kernel = gpflow.kernels.SquaredExponential()
Z = X[::50, :].copy()  # Initialize inducing locations to the first M inputs in the dataset



#m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)
m = SparseVariationalGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N,
                        offset=[np.ones((1, 1)) , np.ones((1, 1)) ])



m.q.offset_x.trainable = False
m.q.offset_y.trainable = False
log_likelihood = tf.function(autograph=False)(m.log_likelihood)

opt = gpflow.optimizers.Scipy()
def objective_closure():
    return - m.elbo(data)

print(m.elbo(data))

opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=100))

print_summary(m)

## generate test points for prediction
xx = np.linspace(xmin - 0.1, xmax + 0.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, 'kx', mew=2)
plt.plot(xx, mean, 'C0', lw=2)
plt.fill_between(xx[:,0],
                 mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                 mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                 color='C0', alpha=0.2)

plt.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.xlim(xmin - 0.1, xmax + 0.1);
plt.show()
