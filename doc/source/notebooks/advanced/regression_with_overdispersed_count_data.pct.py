# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regression with over-dispersed count data
#
# This notebook demonstrates non-parametric regression modelling of over-dispersed count data, i.e., when the response variable $Y$ does not have an approximate Gaussian distribution, but is non-negative discrete $Y \in \mathbb{N}^0$ and when overdispersion occurs, i.e., when the mean of the conditional distribution of $Y$ is not equal to its mean.

# %%
import gpflow

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# %matplotlib inline
import matplotlib.pyplot as plt

tfd = tfp.distributions
plt.rcParams["figure.figsize"] = (12, 4)


# %% [markdown]
# The first generate a synthetic data set for demonstration. We sample data using Tensorflow Probability, using an exponential covariance function to parameterize the latent GP.

# %%
def generate_data(n=30, s=20, seed=123):
    X = np.linspace(-1, 1, n).reshape(-1, 1)
    k = gpflow.kernels.Exponential(lengthscales=0.25, variance=2.0)

    f = (
        tfd.MultivariateNormalTriL(np.repeat(0.0, X.shape[0]), tf.linalg.cholesky(k(X, X)))
        .sample(1, seed=seed)
        .numpy()
        .reshape((n, 1))
    )

    y = tfd.NegativeBinomial(logits=f, total_count=s).sample(1, seed=seed).numpy()
    Y = y.reshape((n, 1))

    return Y, X, f


n, s = 100, 10
Y, X, f = generate_data(n, s)
p = tf.math.sigmoid(f)
mu = s * p / (1 - p)

fig = plt.figure()
plt.plot(X, mu, color="darkred", label="latent function value")
plt.plot(X, Y, "kx", mew=1.5, label="observed data")
plt.legend(bbox_to_anchor=(1.0, 0.5))
plt.show()

# %%
train_idxs = sorted(np.random.choice(n, int(n / 2.0), False))
test_idxs = np.setdiff1d(np.arange(n), train_idxs)

# %% [markdown]
# Next, we specify a covariance function, the likelihood, and a variational GP object. Since, we cannot assume to know the covariance function of the latent GP in practice, we choose to use a Matern 3/2 kernel here.

# %%
kernel = gpflow.kernels.Matern32()
likelihood = gpflow.likelihoods.NegativeBinomial()

m = gpflow.models.VGP(data=(X[train_idxs], Y[train_idxs]), kernel=kernel, likelihood=likelihood)

# %%
opt = gpflow.optimizers.Scipy()
_ = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=200))

# %%
Y_hat, Y_hat_var = m.predict_y(X[test_idxs])

plt.plot(X, mu, color="darkred", label="latent function value", alpha=0.5)
plt.plot(X[test_idxs], Y_hat, color="darkblue", label="predictive posterior mean", alpha=0.5)
plt.plot(X[test_idxs], Y_hat + 1.5 * np.sqrt(Y_hat_var), "--", lw=2, color="darkblue", alpha=0.5)
plt.plot(X[test_idxs], Y_hat - 1.5 * np.sqrt(Y_hat_var), "--", lw=2, color="darkblue", alpha=0.5)
plt.plot(X[train_idxs], Y[train_idxs], "kx", mew=1.5, label="observed data")
plt.legend(bbox_to_anchor=(1.0, 0.5))
plt.show()
