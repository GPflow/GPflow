import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import gpflow
from gpflow import set_trainable
from gpflow.likelihoods.seeger_likelihood import MultiStageLikelihood
from gpflow.optimizers import NaturalGradient

eager = True
# Parameters of the fit
maxiter=100
N = 100
X_train = np.linspace(0,1,N).reshape(-1, 1)
num_data, dim = X_train.shape

L = 3

# Create list of kernels for each GP
Y_train = np.random.randint(2,10,size=(N,1)).astype(float)
ind = np.random.permutation(Y_train.shape[0])[:int(N*.2)]
Y_train[ind] = 1.
ind = np.random.permutation(Y_train.shape[0])[:int(N*.9)]
Y_train[ind] = 0.
data = (X_train, Y_train)


var_obs = np.var(Y_train)


kern_list = [gpflow.kernels.Matern32(variance=var_obs, lengthscales=.1) for _ in range(L)]

# Create multi-output kernel from kernel list
ker = gpflow.kernels.SeparateIndependent(kern_list)

num_inducing = N
Z = np.linspace(X_train.min(), X_train.max(), num_inducing).reshape(-1, 1)

# create multi-output inducing variables from Z
inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
    gpflow.inducing_variables.InducingPoints(Z))


likelihood = MultiStageLikelihood(latent_dim=3, observation_dim=1)

model = gpflow.models.SVGP(kernel=ker,
                           likelihood=likelihood,
                           inducing_variable=inducing_variable,
                           num_data=num_data,
                           num_latent_gps=L)

num_grid=500
X_grid = np.linspace(X_train.min(), X_train.max(), num_grid).reshape(-1, 1)


# NatGrads and Adam for SVGP
# Stop Adam from optimizing the variational parameters
set_trainable(model.inducing_variable, False)
set_trainable(model.q_mu, False)
set_trainable(model.q_sqrt, False)
# Create the optimize_tensors for SVGP
natgrad_adam_opt = tf.optimizers.Adam(1e-3)
natgrad_opt = NaturalGradient(gamma=0.5)
variational_params = [(model.q_mu, model.q_sqrt)]

# %% [markdown]
# Let's optimize the models:

# %%
autotune = tf.data.experimental.AUTOTUNE
batch_size = N


model_loss = lambda : -model.elbo(data)



for it in range(1000):
    natgrad_adam_opt.minimize(model_loss, var_list=model.trainable_variables)
    natgrad_opt.minimize(model_loss, var_list=variational_params)
    print(it, model_loss())

    if it % 10 == 0:

        plt.figure()
        mus, vars = model.predict_f(X_grid)
        plt.plot(X_grid, mus)
        plt.plot(X_train, Y_train, 'k.')
        plt.savefig('test_%d.svg'%it)

# %% [markdown]
#