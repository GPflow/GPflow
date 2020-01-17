

import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

plt.style.use('ggplot')

xmin, xmax = 0, 5
N = 500
X1 = np.random.uniform(low=xmin, high=xmax, size=(N,1))
X2 = np.random.uniform(low=xmin, high=xmax, size=(N,1))
X = np.hstack([X1, X2])

F1 = np.cos(X1 * 4)
F2 = np.sin(X2 * 3)
F = np.hstack([F1, F2])
Y = np.sum(F, axis=-1, keepdims=True) + np.random.randn(N, 1)*.1
Y = np.prod(F, axis=-1, keepdims=True) + np.random.randn(N, 1)*.1
data = (X, Y)

cols = 'rgb'
C = 2

# ## Building the model

kernels = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.SquaredExponential()
]

indices = [slice(0,1), slice(1,2)]

Zs = [X1[::10].copy(), X2[::10].copy()]


m = gpflow.models.MultiplicativeSVGPs(kernels, gpflow.likelihoods.Gaussian(), Zs, indices, num_data=N,
                                deterministic_optimisation=False)

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

maxiter = 10000

logf = run_adam(m, maxiter)


f_means, f_vars = m.predict_fs(X)
p_mean, _ = m.predict_predictor(X)

fig, axarr = plt.subplots(1,3)
ax = axarr[0]
ax.plot(np.arange(maxiter)[::10], logf)
ax.set_xlabel('iteration')
ax.set_ylabel('ELBO');

ax = axarr[1]
for c in range(C):
    o = np.argsort(X[:, c])
    ax.plot(X[o, c], F[o, c], '-', color=cols[c], mew=2);
    ax.plot(X[:, c], f_means[:, c], '.', color=cols[c], mew=2);

ax = axarr[2]
ax.plot(p_mean, Y, 'b*')
ax.plot(Y, Y, 'k-', alpha=.1)
ax.set_ylabel('data')
ax.set_xlabel('prediction');

plt.show()
#
# opt = gpflow.optimizers.Scipy()
#
# def objective_closure():
#     return - m.elbo(data)
#
# mus, vars = m.predict_fs(X)
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
# opt_logs = opt.minimize(objective_closure,
#                         m.trainable_variables,
#                         options=dict(maxiter=1000))

print_summary(m)




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
