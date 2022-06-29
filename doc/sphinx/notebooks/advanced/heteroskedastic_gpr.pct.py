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

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gf
from gpflow import Parameter
from gpflow.utilities import print_summary

np.random.seed(0)
tf.random.set_seed(0)

# %% [markdown]
# ### Create and plot some heteroskedastic data

# %%
N = 101
X = np.linspace(0, 1, N)[:, None]

rand_samples = np.random.normal(0.0, 1.0, size=N)[:, None]
noise = rand_samples * (0.25 + 4 * (1 - X))
signal = 10 * np.sin(2 * np.pi * X)
Y = signal + noise


def plot_distribution(X, Y, mean=None, std=None):
    plt.figure(figsize=(15, 5))
    if mean is not None:
        x = X.squeeze()
        for k in (1, 2):
            lb = (mean - k * std).squeeze()
            ub = (mean + k * std).squeeze()
            plt.fill_between(x, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3)
        plt.plot(x, lb, color="silver")
        plt.plot(x, ub, color="silver")
        plt.plot(X, mean, color="black")
    plt.scatter(X, Y, color="gray", alpha=0.8)
    plt.xlim([0, 1])
    plt.show()


plot_distribution(X, Y)


# %% [markdown]
# ## Train a standard GPR model

# %%
kernel = gf.kernels.Matern52()
base_model = gf.models.SGPR(data=(X, Y), kernel=kernel, inducing_variable=X[:3, :])
base_model.likelihood.variance.assign(0.1)

base_model.maximum_log_likelihood_objective()


opt = gf.optimizers.Scipy()
opt_logs = opt.minimize(
    base_model.training_loss, base_model.trainable_variables, options=dict(maxiter=100)
)
print_summary(base_model)

# %% [markdown]
# This model struggles to describe the data, particularly towards higher values of x where the noise is low.

# %%
mean, var = base_model.predict_y(X)
plot_distribution(X, Y, mean.numpy(), np.sqrt(var.numpy()))

# %% [markdown]
# ### Using a heteroscedastic likelihood

# %% [markdown]
# Some of the GPflow likelihoods allow you to provide a function instead of a single parameter. We can set a linear scale for the Gaussian likelihood.

# %%
kernel = gf.kernels.Matern52()
likelihood = gf.likelihoods.Gaussian(scale=gf.functions.Linear())
het_model = gf.models.GPR(data=(X, Y), kernel=kernel, mean_function=None, likelihood=likelihood)
opt = gf.optimizers.Scipy()
opt_logs = opt.minimize(
    het_model.training_loss, het_model.trainable_variables, options=dict(maxiter=100)
)
print_summary(het_model)

# %%
mean, var = het_model.predict_y(X)
plot_distribution(X, Y, mean.numpy(), np.sqrt(var.numpy()))

# %% [markdown]
# Next we can try a more complex quadratic noise model

# %%
N = 600
# Build inputs X
rand_samples = np.random.normal(0.0, 1.0, size=N)[:, None]
X2 = np.linspace(0, 1, N)[:, None]
signal = (X2 - 0.5) ** 2 + 0.05
Y2 = signal * (1 + 0.3 * rand_samples)

plot_distribution(X2, Y2)

# %%
kernel = gf.kernels.Polynomial(degree=2)
base_model = gf.models.GPR(data=(X2, Y2), kernel=kernel, mean_function=None)
opt = gf.optimizers.Scipy()
opt_logs = opt.minimize(
    base_model.training_loss, base_model.trainable_variables, options=dict(maxiter=100)
)
print_summary(base_model)

# %% [markdown]
# Once again we see that the assumption of a constant noise level is detrimental to the model performance.

# %%
mean, var = base_model.predict_y(X2)
plot_distribution(X2, Y2, mean.numpy(), np.sqrt(var.numpy()))


# %% [markdown]
# GPflow does provie a generic `Polynomial` function we could use here, but for the sake of exercise we will write a custom quadratic here:

# %%
class QuadraticFunction(gf.functions.Function):
    """ Does what it says on the tin.  """

    def __init__(self):
        self.w = Parameter([1.0, 0.0, 0.0])

    def __call__(self, X):
        powers = tf.range(3, dtype=self.w.dtype)
        raised = tf.pow(X, powers)
        return tf.reduce_sum(raised * self.w, axis=-1, keepdims=True)


# %% [markdown]
# Build and train the model

# %%
kernel = gf.kernels.Polynomial(degree=2)
likelihood = gf.likelihoods.Gaussian(scale=QuadraticFunction())
quad_model = gf.models.GPR(data=(X2, Y2), kernel=kernel, mean_function=None, likelihood=likelihood)
opt = gf.optimizers.Scipy()
opt_logs = opt.minimize(
    quad_model.training_loss, quad_model.trainable_variables, options=dict(maxiter=100)
)
print_summary(quad_model)

# %%
mean, var = quad_model.predict_y(X2)
plot_distribution(X2, Y2, mean.numpy(), np.sqrt(var.numpy()))

# %% [markdown]
# ## Further reading
#
# See [Kernel Identification Through Transformers](https://arxiv.org/abs/2106.08185) by Simpson et al.
