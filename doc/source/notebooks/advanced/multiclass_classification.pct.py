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
# # Multiclass classification

# %% [markdown]
# The multiclass classification problem is a regression problem from an input $x \in {\cal X}$ to discrete labels $y\in {\cal Y}$, where ${\cal Y}$ is a discrete set of size $C$ bigger than two (for $C=2$ it is the more usual binary classification).
#
# Labels are encoded in a one-hot fashion, that is if $C=4$ and $y=2$, we note $\bar{y} = [0,1,0,0]$.
#
# The generative model for this problem consists of:
#
# * $C$ latent functions $\mathbf{f} = [f_1,...,f_C]$ with an independent Gaussian Process prior
# * a deterministic function that builds a discrete distribution $\pi(\mathbf{f}) = [\pi_1(f_1),...,\pi_C(f_C)]$ from the latents such that $\sum_c \pi_c(f_c) = 1$
# * a discrete likelihood $p(y|\mathbf{f}) = Discrete(y;\pi(\mathbf{f})) = \prod_c \pi_c(f_c)^{\bar{y}_c}$
#
# A typical example of $\pi$ is the softmax function:
#
# \begin{equation}
# \pi_c (f_c) \propto \exp( f_c)
# \end{equation}
#
# Another convenient one is the robust max:
# \begin{equation}
# \pi_c(\mathbf{f}) = \begin{cases} 1 - \epsilon, & \mbox{if } c = \arg \max_c f_c \\
#  \epsilon /(C-1), & \mbox{ otherwise} \end{cases}
# \end{equation}
#
#
#
#

# %%
import numpy as np
import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

import matplotlib.pyplot as plt

# %matplotlib inline

import gpflow

from gpflow.utilities import print_summary, set_trainable
from gpflow.ci_utils import ci_niter

from multiclass_classification import plot_posterior_predictions, colors

# reproducibility:
np.random.seed(0)
tf.random.set_seed(123)

# %% [markdown]
# ## Sampling from the GP multiclass generative model

# %% [markdown]
# ### Declaring model parameters and input

# %%
# Number of functions and number of data points
C = 3
N = 100

# Lengthscale of the SquaredExponential kernel (isotropic -- change to `[0.1] * C` for ARD)
lengthscales = 0.1

# Jitter
jitter_eye = np.eye(N) * 1e-6

# Input
X = np.random.rand(N, 1)

# %% [markdown]
# ### Sampling

# %%
# SquaredExponential kernel matrix
kernel_se = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
K = kernel_se(X) + jitter_eye

# Latents prior sample
f = np.random.multivariate_normal(mean=np.zeros(N), cov=K, size=(C)).T

# Hard max observation
Y = np.argmax(f, 1).reshape(-1,).astype(int)

# One-hot encoding
Y_hot = np.zeros((N, C), dtype=bool)
Y_hot[np.arange(N), Y] = 1

data = (X, Y)

# %% [markdown]
# ### Plotting

# %%
plt.figure(figsize=(12, 6))
order = np.argsort(X.reshape(-1,))

for c in range(C):
    plt.plot(X[order], f[order, c], ".", color=colors[c], label=str(c))
    plt.plot(X[order], Y_hot[order, c], "-", color=colors[c])


plt.legend()
plt.xlabel("$X$")
plt.ylabel("Latent (dots) and one-hot labels (lines)")
plt.title("Sample from the joint $p(Y, \mathbf{f})$")
plt.grid()
plt.show()

# %% [markdown]
# ## Inference
#

# %% [markdown]
# Inference here consists of computing the posterior distribution over the latent functions given the data $p(\mathbf{f}|Y, X)$.
#
# You can use different inference methods. Here we perform variational inference.
# For a treatment of the multiclass classification problem using MCMC sampling, see [Markov Chain Monte Carlo (MCMC)](../advanced/mcmc.ipynb).
#
#

# %% [markdown]
# ### Approximate inference: Sparse Variational Gaussian Process

# %% [markdown]
# #### Declaring the SVGP model (see [GPs for big data](../advanced/gps_for_big_data.ipynb))

# %%
# sum kernel: Matern32 + White
kernel = gpflow.kernels.Matern32() + gpflow.kernels.White(variance=0.01)

# Robustmax Multiclass Likelihood
invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
likelihood = gpflow.likelihoods.MultiClass(3, invlink=invlink)  # Multiclass likelihood
Z = X[::5].copy()  # inducing inputs

m = gpflow.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=Z,
    num_latent_gps=C,
    whiten=True,
    q_diag=True,
)

# Only train the variational parameters
set_trainable(m.kernel.kernels[1].variance, False)
set_trainable(m.inducing_variable, False)
print_summary(m, fmt="notebook")

# %% [markdown]
# #### Running inference

# %%
opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(
    m.training_loss_closure(data), m.trainable_variables, options=dict(maxiter=ci_niter(1000))
)
print_summary(m, fmt="notebook")

# %%
plot_posterior_predictions(m, X, Y)
