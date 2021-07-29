import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow as gf


from gpflow.utilities import print_summary

N = 101

np.random.seed(0)
tf.random.set_seed(0)

# Build inputs X
X = np.linspace(0, 1, N)[:, None]

# Create outputs Y which includes heteroscedastic noise
rand_samples = np.random.normal(0.0, 1.0, size=N)[:, None]
noise = rand_samples * (0.05 + 0.75 * X)
signal = 2 * np.sin(2 * np.pi * X)

Y = 5*(signal + noise)

# %% [markdown]
# ### Plot Data
# Note how the distribution density (shaded area) and the outputs $Y$ both change depending on the input $X$.

# %%
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
    plt.show()
    plt.close()


# plot_distribution(X, Y)


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
print_summary(model.posterior())

## predict mean and variance of latent GP at test points
mean, var = model.predict_y(X)
plot_distribution(X, Y, mean.numpy(), np.sqrt(var.numpy()))

# fmean, fvar = model.predict_f(X)
# plot_distribution(X, Y, fmean.numpy(), np.sqrt(fvar.numpy()))

## Repeat for standard GPR
base_model = gf.models.GPR(data=(X, Y), kernel=kernel, mean_function=None)
opt_logs = opt.minimize(base_model.training_loss, base_model.trainable_variables, options=dict(maxiter=100))
print_summary(base_model)

## predict mean and variance of latent GP at test points
mean, var = base_model.predict_y(X)
plot_distribution(X, Y, mean.numpy(), np.sqrt(var.numpy()))




# ## generate test points for prediction
# xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)
#
# ## predict mean and variance of latent GP at test points
# mean, var = model.predict_f(xx)
#
# ## generate 10 samples from posterior
# tf.random.set_seed(1)  # for reproducibility
# samples = model.predict_f_samples(xx, 10)  # shape (10, 100, 1)
#
# ## plot
# plt.figure(figsize=(12, 6))
# plt.plot(X, Y, "kx", mew=2)
# plt.plot(xx, mean, "C0", lw=2)
# plt.fill_between(
#     xx[:, 0],
#     mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
#     mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
#     color="C0",
#     alpha=0.2,
# )
#
# plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
# _ = plt.xlim(-0.1, 1.1)


# %% [markdown]
# ## Further reading
#
# See [Kernel Identification Through Transformers](https://arxiv.org/abs/2106.08185) by Simpson et al.
