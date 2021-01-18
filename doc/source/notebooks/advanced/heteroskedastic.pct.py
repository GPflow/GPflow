# ---
# jupyter:
#   jupytext:
#     formats: ipynb,.pct.py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Heteroskedastic Likelihood and Multi-Latent GP

# %% [markdown]
# ## Standard (Homoskedastic) Regression
# In standard GP regression, the GP latent function is used to learn the location parameter of a likelihood distribution (usually a Gaussian) as a function of the input $x$, whereas the scale parameter is considered constant. This is a homoskedastic model, which is unable to capture variations of the noise distribution with the input $x$.
#
#
# ## Heteroskedastic Regression
# This notebooks shows how to construct a model which uses multiple (here two) GP latent functions to learn both the location and the scale of the Gaussian likelihood distribution. It does so by connecting a **Multi-Output Kernel**, which generates multiple GP latent functions, to a **Heteroskedastic Likelihood**, which maps the latent GPs into a single likelihood.
#
# The generative model is described as:
#
# $$ f_1(x) \sim \mathcal{GP}(0, k_1(\cdot, \cdot)) $$
# $$ f_2(x) \sim \mathcal{GP}(0, k_2(\cdot, \cdot)) $$
# $$ \text{loc}(x) = f_1(x) $$
# $$ \text{scale}(x) = \text{transform}(f_2(x)) $$
# $$ y_i|f_1, f_2, x_i \sim \mathcal{N}(\text{loc}(x_i),\;\text{scale}(x_i)^2)$$
#
# The function $\text{transform}$ is used to map from the unconstrained GP $f_2$ to **positive-only values**, which is required as it represents the $\text{scale}$ of a Gaussian likelihood. In this notebook, the $\exp$ function will be used as the $\text{transform}$. Other positive transforms such as the $\text{softplus}$ function can also be used.

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf


# %% [markdown]
# ## Data Generation
# We generate heteroskedastic data by substituting the random latent functions $f_1$ and $f_2$ of the generative model by deterministic $\sin$ and $\cos$ functions. The input $X$ is built with $N=1001$ uniformly spaced values in the interval $[0, 4\pi]$. The outputs $Y$ are still sampled from a Gaussian likelihood.
#
# $$ x_i \in [0, 4\pi], \quad i = 1,\dots,N $$
# $$ f_1(x) = \sin(x) $$
# $$ f_2(x) = \cos(x) $$
# $$ \text{loc}(x) = f_1(x) $$
# $$ \text{scale}(x) = \exp(f_2(x)) $$
# $$ y_i|x_i \sim \mathcal{N}(\text{loc}(x_i),\;\text{scale}(x_i)^2)$$

# %%
N = 1001

np.random.seed(0)
tf.random.set_seed(0)

# Build inputs X
X = np.linspace(0, 4 * np.pi, N)[:, None]  # X must be of shape [N, 1]

# Deterministic functions in place of latent ones
f1 = np.sin
f2 = np.cos

# Use transform = exp to ensure positive-only scale values
transform = np.exp

# Compute loc and scale as functions of input X
loc = f1(X)
scale = transform(f2(X))

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

# %%
likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
    distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
    scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
)

print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")

# %% [markdown]
# ### Kernel
# This implements the following part of the generative model:
# $$ f_1(x) \sim \mathcal{GP}(0, k_1(\cdot, \cdot)) $$
# $$ f_2(x) \sim \mathcal{GP}(0, k_2(\cdot, \cdot)) $$
# with both kernels being modeled as separate and independent $\text{SquaredExponential}$ kernels.

# %%
kernel = gpf.kernels.SeparateIndependent(
    [
        gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
        gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ]
)
# The number of kernels contained in gpf.kernels.SeparateIndependent must be the same as likelihood.latent_dim

# %% [markdown]
# ### Inducing Points
# Since we will use the **SVGP** model to perform inference, we need to implement the inducing variables $U_1$ and $U_2$, both with size $M=20$, which are used to approximate $f_1$ and $f_2$ respectively, and initialize the inducing points positions $Z_1$ and $Z_2$. This gives a total of $2M=40$ inducing variables and inducing points.
#
# The inducing variables and their corresponding inputs will be Separate and Independent, but both $Z_1$ and $Z_2$ will be initialized as $Z$, which are placed as $M=20$ equally spaced points in $[\min(X), \max(X)]$.
#

# %%
M = 20  # Number of inducing variables for each f_i

# Initial inducing points position Z
Z = np.linspace(X.min(), X.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
    [
        gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
        gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
    ]
)

# %% [markdown]
# ### SVGP Model
# Build the **SVGP** model by composing the **Kernel**, the **Likelihood** and the **Inducing Variables**.
#
# Note that the model needs to be instructed about the number of latent GPs by passing `num_latent_gps=likelihood.latent_dim`.

# %%
model = gpf.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_latent_gps=likelihood.latent_dim,
)

model

# %% [markdown]
# ## Model Optimization
#
# ### Build Optimizers (NatGrad + Adam)

# %%
data = (X, Y)
loss_fn = model.training_loss_closure(data)

gpf.utilities.set_trainable(model.q_mu, False)
gpf.utilities.set_trainable(model.q_sqrt, False)

variational_vars = [(model.q_mu, model.q_sqrt)]
natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

adam_vars = model.trainable_variables
adam_opt = tf.optimizers.Adam(0.01)


@tf.function
def optimisation_step():
    natgrad_opt.minimize(loss_fn, variational_vars)
    adam_opt.minimize(loss_fn, adam_vars)


# %% [markdown]
# ### Run Optimization Loop

# %%
epochs = 100
log_freq = 20

for epoch in range(1, epochs + 1):
    optimisation_step()

    # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
    if epoch % log_freq == 0 and epoch > 0:
        print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
        Ymean, Yvar = model.predict_y(X)
        Ymean = Ymean.numpy().squeeze()
        Ystd = tf.sqrt(Yvar).numpy().squeeze()
        plot_distribution(X, Y, Ymean, Ystd)

model

# %% [markdown]
# ## The conditional output distribution
#
# Here we show how to get the conditional output distribution and plot samples from it. In order to plot the uncertainty associated with that, we also get the percentiles from the conditional output distribution.
# Although the likelihood is Gaussian, the marginal posterior is not hence plotting with just the mean and variance would be misrepresentative of the real situation.

# %%
plot_distribution(X, Y, Ymean, Ystd)

y_dist = model.conditional_y_dist(X)
samples = y_dist.sample(10_000)

# The folling is equivalent at doing:
# y_lo_lo, y_lo, y_hi, y_hi_hi = np.quantile(samples, q=(0.025, 0.159, 0.841, 0.975), axis=0)
# Note how, contrary to the binary classification case, here we get the percentiles directly from the 
# conditional output distribution
y_lo_lo, y_lo, y_hi, y_hi_hi = y_dist.y_percentile(p=(2.5, 15.9, 84.1, 97.5), num_samples=10_000)

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X, np.mean(samples, axis=0), c='k')
ax.fill_between(X.squeeze(), y_lo[...,0], y_hi[...,0], color="silver", alpha=1 - 0.05 * 1 ** 3)
ax.fill_between(X.squeeze(), y_lo_lo[...,0], y_hi_hi[...,0], color="silver", alpha=1 - 0.05 * 2 ** 3)
ax.scatter(X, Y, color="gray", alpha=0.8)

# %%
p_mu_samples, p_var_samples = y_dist.parameter_samples(10_000)
# p_mu_lo, p_mu_hi = np.quantile(p_mu_samples, q=(0.159, 0.841), axis=0)
# p_var_lo, p_var_hi = np.quantile(p_var_samples, q=(0.159, 0.841), axis=0)

(p_mu_lo, p_mu_hi), (p_var_lo, p_var_hi) = y_dist.parameter_percentile(p=(15.9, 84.1), num_samples=10_000)

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.scatter(X, Y, color="gray", alpha=0.8)
ax.plot(X, Ymean, c='k')
ax.plot(X, np.mean(p_mu_samples, axis=0), ls='--')

ax.fill_between(X.squeeze(), p_mu_lo[...,0], p_mu_hi[...,0], color="silver", alpha=0.8)

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X, scale ** 2, c='k')
ax.plot(X, np.mean(p_var_samples, axis=0), ls='--')
ax.fill_between(X.squeeze(), p_var_lo[...,0], p_var_hi[...,0], color="silver", alpha=0.8)

# %%
from scipy.stats import norm
f_mu = 0.0
f_var = 0.1
g_mu = 0.0
g_var = 0.1
n_samples = 1_00_000

f = np.random.normal(loc=f_mu, scale=np.sqrt(f_var), size=n_samples)
g = np.random.normal(loc=g_mu, scale=np.sqrt(g_var), size=n_samples)

y = np.random.normal(loc=f, scale=np.exp(g))

fig, ax = plt.subplots(1, 1)
ax.hist(y, bins=100, density=True, alpha=0.5)
xx = np.linspace(-10, 10, 101)
noise_var = np.exp(g_mu + g_var / 2) ** 2
y_var = (f_var + noise_var) 
ax.plot(xx, norm.pdf(xx, loc=f_mu, scale=y_var ** 0.5), c='k')
ax.set_yscale('log')
ax.set_ylim(1e-8, 1e0)



# %% [markdown]
# ## Further reading
#
# See [Chained Gaussian Processes](http://proceedings.mlr.press/v51/saul16.html) by Saul et al. (AISTATS 2016).
