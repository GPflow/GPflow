# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Heteroskedastic Likelihood for GPR models

# %% [markdown]
# ## Standard (Homoskedastic) Regression
# In standard GP regression, the GP latent function is used to learn the location parameter of a likelihood distribution (usually a Gaussian) as a function of the input $x$, whereas the scale parameter is considered constant. This is a homoskedastic model, which is unable to capture variations of the noise distribution with the input $x$.
#
#
# ## Heteroskedastic Regression
# This notebooks shows how to make use of a noise model where the noise amplitude varies linearly with respect to the input. 
#
# The noise model is described as:
#
# $$ \sigma(x) \propto (X - c) $$
# where $c$ denotes some offset location. 

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gf


# %% [markdown]
# ## Data Generation
# We begin by illustrating the generation of heteroskedastic data from the GPR model


# %%
from gpflow.utilities import print_summary

N = 101

np.random.seed(0)
tf.random.set_seed(0)

# Build inputs X
X = np.linspace(0, 1, N)[:, None]

# Sample outputs Y from Gaussian Likelihood
Y = np.random.normal(loc, scale)

# %% [markdown]
# ### Plot Data
# Note how the distribution density (shaded area) and the outputs $Y$ both change depending on the input $X$.

# %%
def plot_distribution(X, Y, loc, scale):
    plt.figure(figsize=(15, 5))
    x = X.squeeze()
    for k in (1, 2):
        lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        plt.fill_between(x, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3)
    plt.plot(x, lb, color="silver")
    plt.plot(x, ub, color="silver")
    plt.plot(X, loc, color="black")
    plt.scatter(X, Y, color="gray", alpha=0.8)
    plt.show()
    plt.close()


plot_distribution(X, Y, loc, scale)


# %% [markdown]
# ## Build Model

# %% [markdown]
# ### Likelihood
# This implements the following part of the generative model:
# $$ \text{loc}(x) = f_1(x) $$
# $$ \text{scale}(x) = \text{transform}(f_2(x)) $$
# $$ y_i|f_1, f_2, x_i \sim \mathcal{N}(\text{loc}(x_i),\;\text{scale}(x_i)^2)$$

# %% [markdown]
# ### Select a kernel
# %%
kernel = gf.kernels.Matern52()

# %% [markdown]
# ### HeteroskedasticGPR Model
# Build the **GPR** model with the data and kernel

# %%
model = gf.models.het_GPR(data=(X, Y), kernel=kernel, mean_function=None)

# %% [markdown]
# ## Model Optimization proceeds as in the GPR notebook
# %%
opt = gf.optimizers.Scipy()

# %% [markdown]
# %%
opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
print_summary(model)

## generate test points for prediction
xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = model.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = model.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 1.1)


# %% [markdown]
# ## Further reading
#
# See [Kernel Identification Through Transformers](https://arxiv.org/abs/2106.08185) by Simpson et al.
