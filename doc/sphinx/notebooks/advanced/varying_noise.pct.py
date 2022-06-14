# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gaussian process regression with varying output noise

# %% [markdown]
# This notebook shows how to construct a Gaussian process model where different noise is assumed for different data points. The model is:
#
# \begin{align}
# f(\cdot) &\sim \mathcal{GP}\big(0, k(\cdot, \cdot)\big) \\
# y_i | f, x_i &\sim \mathcal N\big(y_i; f(x_i), \sigma^2_i\big)
# \end{align}
#
# We'll demonstrate three methods. In the first demonstration we'll show how to fit the noise variance to a simple function. In the second demonstration we'll assume that the data comes from two different groups and show how to learn separate noise variance for the groups. In the final demonstration we'll assume you have multiple samples at each location $x$ and show how to use the empirical variance.
# variances.

# %%
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gf
from gpflow import Parameter
from gpflow.utilities import print_summary

# %matplotlib inline

optimizer_config = dict(maxiter=reduce_in_tests(100))
n_data = reduce_in_tests(300)
X_plot = np.linspace(0.0, 1.0, reduce_in_tests(101))[:, None]


# %% [markdown]
# To help us later we'll first define a small function for plotting data with predictions:

# %%
def plot_distribution(X, Y, X_plot=None, mean=None, var=None):
    X = X.squeeze(axis=-1)
    Y = Y.squeeze(axis=-1)
    plt.figure(figsize=(15, 5))
    if X_plot is not None:
        X_plot = X_plot.squeeze(axis=-1)
        mean = mean.squeeze(axis=-1)
        var = var.squeeze(axis=-1)
        std = np.sqrt(var)
        for k in (1, 2):
            lb = mean - k * std
            ub = mean + k * std
            plt.fill_between(X_plot, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3)
        plt.plot(X_plot, lb, color="silver")
        plt.plot(X_plot, ub, color="silver")
        plt.plot(X_plot, mean, color="black")
    plt.scatter(X, Y, color="gray", alpha=0.8)
    plt.xlim([0, 1])
    plt.show()


# %% [markdown]
# ## Demo 1: known noise variances
# ### Generate data
# We create a utility function to generate synthetic data, including noise that varies amongst the data:

# %%
def generate_data():
    rng = np.random.default_rng(42)  # for reproducibility
    X = rng.uniform(0.0, 1.0, (n_data, 1))
    signal = (X - 0.5) ** 2 + 0.05
    noise_scale = 0.5 * signal
    noise = noise_scale * rng.standard_normal((n_data, 1))
    Y = signal + noise
    return X, Y


X, Y = generate_data()

# %% [markdown]
# The data alone looks like:

# %%
plot_distribution(X, Y)

# %% [markdown]
# ### Try a naive fit
# If we try to fit a naive GPR to this we get:

# %%
model = gf.models.GPR(
    data=(X, Y),
    kernel=gf.kernels.Matern52(),
)
gf.optimizers.Scipy().minimize(
    model.training_loss, model.trainable_variables, options=optimizer_config
)
print_summary(model)

# %%
mean, var = model.predict_y(X_plot)
plot_distribution(X, Y, X_plot, mean.numpy(), var.numpy())

# %% [markdown]
# Notice how this naive model underestimates the noise near the center of the figure, but overestimates at for small and large values of x?

# %% [markdown]
# ### Fit a polynomial to the noise scale
#
# To fit a model with varying noise we first create a `Likelihood` with a (polynomial) `Function` for the noise,
# then pass that likelihood to our model.

# %%
model = gf.models.GPR(
    data=(X, Y),
    kernel=gf.kernels.Matern52(),
    likelihood=gf.likelihoods.Gaussian(scale=gf.functions.Polynomial(degree=2)),
)
gf.optimizers.Scipy().minimize(
    model.training_loss, model.trainable_variables, options=optimizer_config
)
print_summary(model)

# %%
mean, var = model.predict_y(X_plot)
plot_distribution(X, Y, X_plot, mean.numpy(), var.numpy())


# %% [markdown]
# ## Demo 2: grouped noise variances
#
# In this demo, we won't assume that the noise variances is a function of $x$, but we will assume that they're known to be in two groups. This example represents a case where we might know that data has been collected by two instruments with different fidelity, but we do not know what those fidelities are.
#
# Of course it would be straightforward to add more groups. We'll stick with two for simplicity.

# %% [markdown]
# ### Generate data

# %%
def get_groups(X):
    return np.where((((0.2 < X) & (X < 0.4)) | ((0.7 < X) & (X < 0.8))), 0, 1)


def generate_data():
    rng = np.random.default_rng(42)  # for reproducibility
    X = rng.uniform(0.0, 1.0, (n_data, 1))
    signal = np.sin(6 * X)
    noise_scale = 0.1 + 0.4 * get_groups(X)
    noise = noise_scale * rng.standard_normal((n_data, 1))
    Y = signal + noise
    return X, Y


X, Y = generate_data()

# %% [markdown]
# And here's a plot of the raw data:

# %%
plot_distribution(X, Y)

# %% [markdown]
# ### Data structure
#
# In this case, we need to let the model know which group each data point belongs to, so we'll stack the group with the $x$ data:

# %%
X_data = np.hstack([X, get_groups(X)])
X_plot_data = np.hstack([X_plot, get_groups(X_plot)])

# %% [markdown]
# ### Fit a naive model
#
# Again we'll start by fitting a naive GPR, to demonstrate the problem.
#
# Notice how we need to set the active dimensions of the kernel, for the kernel not to apply to the group column.

# %%
model = gf.models.GPR(
    data=(X_data, Y),
    kernel=gf.kernels.Matern52(active_dims=[0]),
)
gf.optimizers.Scipy().minimize(
    model.training_loss, model.trainable_variables, options=optimizer_config
)
print_summary(model)

# %%
mean, var = model.predict_y(X_plot_data)
plot_distribution(X, Y, X_plot, mean.numpy(), var.numpy())

# %% [markdown]
# ### Use multiple functions for the noise variance
#
# To model this we will create two noise variance functions, each of which are just a constant, and then switch between them depending on the group labels.
#
# Notice that we initialize the function to a positive value. They would have defaulted to `0.0`, but that doesn't work for a variance.

# %%
model = gf.models.GPR(
    data=(X_data, Y),
    kernel=gf.kernels.Matern52(active_dims=[0]),
    likelihood=gf.likelihoods.Gaussian(
        variance=gf.functions.SwitchedFunction(
            [
                gf.functions.Constant(1.0),
                gf.functions.Constant(1.0),
            ]
        )
    ),
)
gf.optimizers.Scipy().minimize(
    model.training_loss, model.trainable_variables, options=optimizer_config
)
print_summary(model)

# %%
mean, var = model.predict_y(X_plot_data)
plot_distribution(X, Y, X_plot, mean.numpy(), var.numpy())

# %% [markdown]
# ## Demo 3: Empirical noise variance
#
# In this demo we will assume that you have multiple measurements at each $x$ location, and want to use the empirical variance at each location.

# %% [markdown]
# ### Generate data

# %%
n_data = reduce_in_tests(20)
n_repeats = reduce_in_tests(4)


def generate_data():
    rng = np.random.default_rng(42)  # for reproducibility
    X = rng.uniform(0.0, 1.0, (n_data, 1))
    signal = np.sin(6 * X)
    noise_scale = rng.uniform(0.1, 1.0, (n_data, 1))
    noise = noise_scale * rng.standard_normal((n_data, n_repeats))
    Y = signal + noise
    return X, Y


X, Y = generate_data()
Y_mean = np.mean(Y, axis=-1, keepdims=True)
Y_var = np.var(Y, axis=-1, keepdims=True)

# %% [markdown]
# For the sake of plotting we'll create flat lists of (x, y) pairs:

# %%
X_flat = np.broadcast_to(X, Y.shape).reshape((-1, 1))
Y_flat = Y.reshape((-1, 1))
plot_distribution(X_flat, Y_flat)

# %% [markdown]
# ### Data structure
#
# Our GPs don't like it when multiple data points occupy the same $x$ location. So we'll reduce this dataset to the means and the variances at each $x$, and use a custom function to model inject the empirical variances into the model.

# %%
Y_mean = np.mean(Y, axis=-1, keepdims=True)

# %% [markdown]
# ### Fit a naive model
#
# Again we'll start by fitting a naive GPR, to demonstrate the problem.
#
# Notice:
# 1. We cannot actually model the point-wise variance. We just model `Y_mean`.
# 2. Similarly we cannot actually plot the noise variance. In the other section we were plotting `predict_y` - here we plot `predict_x`.

# %%
model = gf.models.GPR(
    data=(X, Y_mean),
    kernel=gf.kernels.Matern52(),
)
gf.optimizers.Scipy().minimize(
    model.training_loss, model.trainable_variables, options=optimizer_config
)
print_summary(model)

# %%
mean, var = model.predict_f(X_plot)
plot_distribution(X_flat, Y_flat, X_plot, mean.numpy(), var.numpy())


# %% [markdown]
# ### Create custom function for the noise variance
#
# We're modelling the `Y_mean` and its standard error is `Y_var / n_repeats`. We will create a simple function that ignores its input and returns these values as a constant. This will obviously only work for the `X` that corresponds to the `Y` we computed the variance of - that is good enough for model training, but notice that it will not allow us to use `predict_y`.

# %%
class FixedVarianceOfMean(gf.functions.Function):
    def __init__(self, Y):
        n_repeats = Y.shape[-1]
        self.var_mean = np.var(Y, axis=-1, keepdims=True) / n_repeats

    def __call__(self, X):
        return self.var_mean


# %% [markdown]
# Now we can plug that into our model:

# %%
model = gf.models.GPR(
    data=(X, Y_mean),
    kernel=gf.kernels.Matern52(active_dims=[0]),
    likelihood=gf.likelihoods.Gaussian(variance=FixedVarianceOfMean(Y)),
)
gf.optimizers.Scipy().minimize(
    model.training_loss, model.trainable_variables, options=optimizer_config
)
print_summary(model)

# %%
mean, var = model.predict_f(X_plot)
plot_distribution(X_flat, Y_flat, X_plot, mean.numpy(), var.numpy())

# %% [markdown]
# ## Further reading
# To model the variance using a second GP, see the [Heteroskedastic regression notebook](heteroskedastic.ipynb).
