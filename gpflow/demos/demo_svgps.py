

import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

plt.style.use('ggplot')

xmin, xmax = 0, 5
N = 2000
X1 = np.random.uniform(low=xmin, high=xmax, size=(N,1))
X2 = np.random.uniform(low=xmin, high=xmax, size=(N,1))
X = np.hstack([X1, X2])

F1 = np.cos(X1 * 4)
F2 = np.sin(X2 * 3)
F = np.hstack([F1, F2])
Y = np.sum(F, axis=-1, keepdims=True) + np.random.randn(N, 1)*.5
data = (X, Y)

cols = 'rgb'
C = 2

# ## Building the model

kernels = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.SquaredExponential()
]

Zs = [X1[::50].copy(), X2[::50].copy()]

m = gpflow.models.SVGPs(kernels, gpflow.likelihoods.Gaussian(), Zs, num_data=N)

log_likelihood = tf.function(autograph=False)(m.log_likelihood)

opt = gpflow.optimizers.Scipy()

def objective_closure():
    return - m.elbo(data)

mus, vars = m.predict_fs(X)

print(m.elbo(data))
print(m.num_components)
print(m.predict_fs(X)[0].shape)
print(m.predict_fs(X)[1].shape)

with tf.GradientTape() as tape:
    tape.watch(m.trainable_variables)
    res = m.elbo(data)
grads = tape.gradient(res, m.trainable_variables)
for v, g in zip(m.trainable_variables, grads):
    if g is None:
        print(v, g)

opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=1000))

print_summary(m)

f_means, f_vars = m.predict_fs(X)
print(f_means.shape)
for c in range(C):
    o = np.argsort(X[:, c])
    plt.plot(X[o, c], F[o, c], '-', color=cols[c], mew=2);
    plt.plot(X[:, c], f_means[:, c], '.', color=cols[c], mew=2);

plt.show()


# ## generate test points for prediction
# xx = np.linspace(xmin - 0.1, xmax + 0.1, 100).reshape(100, 1)  # test points must be of shape (N, D)
#
# ## predict mean and variance of latent GP at test points
# mean, var = m.predict_f(xx)
#
# ## generate 10 samples from posterior
# samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)
#
# ## plot
# plt.figure(figsize=(12, 6))
# plt.plot(X, Y, 'kx', mew=2)
# plt.plot(xx, mean, 'C0', lw=2)
# plt.fill_between(xx[:,0],
#                  mean[:,0] - 1.96 * np.sqrt(var[:,0]),
#                  mean[:,0] + 1.96 * np.sqrt(var[:,0]),
#                  color='C0', alpha=0.2)
#
# plt.plot(xx, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
# plt.xlim(xmin - 0.1, xmax + 0.1);
# plt.show()
