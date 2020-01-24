

import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
np.random.seed(0)

plt.style.use('ggplot')



import pandas as pd

# #=================================================================================================
#
# # rho ~ f1(x1)  + f2(x2) * f3(x3)
# # paradigm parameters
#
# # --------------- model
xrange1 = [0, 2]
f1 = lambda x: np.exp(x / 2) - 1.5
xrange2 = [0, np.pi]
f2 = lambda x: 1 + np.cos(2.*x + np.pi / 3)
xrange3 =[0, 2]
f3 = lambda x: -np.sin(x)
fs = [f1, f2, f3]
xranges = [xrange1, xrange2, xrange3]
C = len(fs)
#
# # --------------- observations
# n = 500  # number of observations
# observations = 'poisson'
# assert observations in ['poisson', 'binomial', 'gaussian']
# # generate input variables (from uniform distribution in defined range)
# X = np.zeros((n, C))
# for c in range(3):
#     X[:, c] = xranges[c][0] + np.random.rand(n,) * np.diff(xranges[c])
# # generate predictor
# rho = (f1(X[:, 0]) + f2(X[:, 1]) * f3(X[:, 2]))[..., None] # predictor
# # store individual contributions
# F = np.vstack([fs[c](X[:, c]) for c in range(C)]).T
# # generate observations
# phi = lambda x:1. / (1 + np.exp(-x))
# if observations == 'binomial':
#     pval = phi(rho)  # pass through sigmoid
#     Y = (pval > np.random.rand(n, 1)).astype(float)  # generate response from Bernoulli distribution
# elif observations == 'poisson':
#     Y = np.random.poisson(np.exp(rho)).astype(float)
# elif observations == 'gaussian':
#     Y = np.random.randn(*rho.shape) * .5 + rho
# N, C = X.shape
# data = (X, Y)
# print(X.shape, Y.shape)

n = 500
observations = 'poisson'
XFY = pd.read_csv('data.csv').to_numpy()
X, F, Y = XFY[:n, slice(0,3)], XFY[:n, slice(3,6)], XFY[:n, slice(6,9)]
rho = (f1(X[:, 0]) + f2(X[:, 1]) * f3(X[:, 2]))[..., None] # predictor
F = np.vstack([fs[c](X[:, c]) for c in range(C)]).T

N = len(Y)
data = (X, Y)
#=================================================================================================


cols = 'rgb'
#C = 2

# ## Building the model



from gpflow.kernels import SquaredExponential, ConditionedKernel
kernels = [
    SquaredExponential(),
    SquaredExponential(),
    SquaredExponential()
]



from gpflow.mean_functions import Zero
mean_functions = [Zero(), Zero(), Zero()]

indices = [slice(c,c+1) for c in range(C)]

M = 10
Zs = [
    np.linspace(X[:, 0].min(), X[:, 0].max(), M).reshape(-1, 1).copy(),
    np.linspace(X[:, 1].min(), X[:, 1].max(), M).reshape(-1, 1).copy(),
    np.linspace(X[:, 2].min(), X[:, 2].max(), M).reshape(-1, 1).copy()
]


q_mus = [
    np.zeros((len(Zs[0]), 1)),
    np.zeros((len(Zs[1]), 1)),
    np.zeros((len(Zs[2]), 1)),
]

if observations == 'binomial':
    likelihood = gpflow.likelihoods.Bernoulli()
elif observations == 'poisson':
    likelihood = gpflow.likelihoods.Poisson()
elif observations == 'gaussian':
    likelihood = gpflow.likelihoods.Gaussian(variance=.1)


offsets_x = [
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    np.zeros((1, 1))
]
offsets_y = [
    np.ones((1, 1))*-.5,
    np.ones((1, 1))*0.,
    np.ones((1, 1))*0.
]


m = gpflow.models.Custom(kernels, likelihood, Zs, indices, num_data=N,
                        deterministic_optimisation=False,
                         mean_functions=mean_functions,
                         offsets_x=offsets_x,
                         offsets_y=offsets_y, num_samples=10)
for o in m.offsets_x:
    o.trainable = False

m.offsets_y[0].trainable = False
m.offsets_y[2].trainable = False

print(m.offsets_x)

#m.offsets.trainable = False

log_likelihood = tf.function(autograph=False)(m.log_likelihood)

# We turn off training for inducing point locations
gpflow.utilities.set_trainable(m.inducing_variables, False)


@tf.function(autograph=False)
def optimization_step(optimizer, model, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = - m.elbo(batch)
        grads = tape.gradient(objective, m.trainable_variables)
    optimizer.apply_gradients(zip(grads, m.trainable_variables))
    return objective


def run_adam(model, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    adam = tf.optimizers.Adam(learning_rate=1e-3)
    for step in range(iterations):
        elbo = - optimization_step(adam, model, data)
        if step % 10 == 0:
            logf.append(elbo.numpy())
            print(step, elbo.numpy())
    return logf

maxiter = 5000

logf = run_adam(m, maxiter)


f_means, f_vars = m.predict_fs(X)
p_mean, p_var = m.predict_predictor(X)

f_means, f_vars = f_means.numpy(), f_vars.numpy()
f_stds = np.sqrt(f_vars)

fig, axarr = plt.subplots(3, 1, figsize= (5,15))
axarr = axarr.flatten()
ax = axarr[0]
ax.plot(np.arange(maxiter)[::10], logf)
ax.set_xlabel('iteration')
ax.set_ylabel('ELBO');

ax = axarr[1]
for c in range(C):
    o = np.argsort(X[:, c])
    ax.plot(X[o, c], F[o, c], '--', color=cols[c], mew=2)
    ax.fill_between(X[o, c],
                    f_means[o, c] - 2. * f_stds[o, c],
                    f_means[o, c] + 2. * f_stds[o, c], alpha=.2, facecolor=cols[c])
    ax.plot(X[o, c], f_means[o, c], '-', color=cols[c], mew=2)

print(m.inducing_variables[0].Z)

ax = axarr[2]
ax.errorbar(rho, p_mean, yerr=np.sqrt(p_var), color='b', fmt='o')
ax.plot(rho, rho, 'k-', alpha=.1)
ax.set_xlabel('data rho')
ax.set_ylabel('prediction');

plt.suptitle('N=%d - %s' %(n, observations))
plt.savefig("%s_%d.pdf"%(observations, n))
plt.show()

