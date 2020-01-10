

import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

plt.style.use('ggplot')


xmin, xmax = 0, 5
N = 500
X1 = np.linspace(xmin, xmax, N).reshape(-1,1)
X = np.hstack([X1, X1, X1])

f1 = lambda x: np.cos(x * 4)
f2 = lambda x: np.exp(- (x-1)**2 * 2.)*3

NY = 3
new_years = [1, 2, 3]

Y = np.zeros((N, NY))
for ny in range(NY):
    F = f1(X1) + f2(X1 - new_years[ny])
    Y[:, ny] = F.flatten()

# stacked
Y += np.random.randn(N, NY)*.5

# unstack
Y = Y.reshape(-1, 1)


data = (X, Y)

cols = 'rgb'
C = 2

# ## Building the model

kernels = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.SquaredExponential()
]

Zs = [X1[::20].copy(), X1[::20].copy()]

m = gpflow.models.CNY(kernels, gpflow.likelihoods.Gaussian(), Zs, num_data=N, new_years=new_years)

log_likelihood = tf.function(autograph=False)(m.log_likelihood)

opt = gpflow.optimizers.Scipy()

def objective_closure():
    return - m.elbo(data)

print(m.num_components)
print(m.num_years)

print(m.elbo(data))

opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=1000))

print(m.elbo(data))

#
mus, vars = m.predict_fs(X)


plt.plot(X1, mus[:, 0:1], 'r')
plt.plot(X1, mus[:, 1:2], 'b')
plt.show()


M, S = m.predict_years(X)
fig, axarr = plt.subplots(3,1)
for ny in range(len(new_years)):
    axarr[ny].plot(X, M[:, ny:ny+1])
plt.show()
#
# print(m.elbo(data))
# print(m.num_components)
# print(m.predict_fs(X)[0].shape)
# print(m.predict_fs(X)[1].shape)
#
# with tf.GradientTape() as tape:
#     tape.watch(m.trainable_variables)
#     res = m.elbo(data)
# grads = tape.gradient(res, m.trainable_variables)
# for v, g in zip(m.trainable_variables, grads):
#     if g is None:
#         print(v, g)
#

#
# print_summary(m)
#
# f_means, f_vars = m.predict_fs(X)
# print(f_means.shape)
# for c in range(C):
#     o = np.argsort(X[:, c])
#     plt.plot(X[o, c], F[o, c], '-', color=cols[c], mew=2);
#     plt.plot(X[:, c], f_means[:, c], '.', color=cols[c], mew=2);
#
# plt.show()


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
