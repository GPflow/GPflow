# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="eYrSpUncKGSk"
# # Faster predictions by caching

# + [markdown] id="PLuPjfS7KLQ-"
# The default behaviour of `predict_f` in GPflow models is to compute the predictions from scratch on each call. This is convenient when predicting and training are interleaved, and simplifies the use of these models. There are some use cases, such as Bayesian optimisation, where prediction (at different test points) happens much more frequently than training. In these cases it is convenient to cache parts of the calculation which do not depend upon the test points, and reuse those parts between predictions.
#
# There are three models to which we want to add this caching capability: GPR, (S)VGP and SGPR. The VGP and SVGP can be considered together; the difference between the models is whether to condition on the full training data set (VGP) or on the inducing variables (SVGP).

# + [markdown] id="EACkO-iRKM5T"
# ## Posterior predictive distribution
#
# The posterior predictive distribution evaluated at a set of test points $\mathbf{x}_*$ for a Gaussian process model is given by:
# \begin{equation*}
# p(\mathbf{f}_*|X, Y) = \mathcal{N}(\mu, \Sigma)
# \end{equation*}
#
# In the case of the GPR model, the parameters $\mu$ and $\Sigma$ are given by:
# \begin{equation*}
# \mu = K_{nm}[K_{mm} + \sigma^2I]^{-1}\mathbf{y}
# \end{equation*}
# and
# \begin{equation*}
# \Sigma = K_{nn} - K_{nm}[K_{mm} + \sigma^2I]^{-1}K_{mn}
# \end{equation*}
#
# The posterior predictive distribution for the VGP and SVGP model is parameterised as follows:
# \begin{equation*}
# \mu = K_{nu}K_{uu}^{-1}\mathbf{u}
# \end{equation*}
# and
# \begin{equation*}
# \Sigma = K_{nn} - K_{nu}K_{uu}^{-1}K_{un}
# \end{equation*}
#
# Finally, the parameters for the SGPR model are:
# \begin{equation*}
# \mu = K_{nu}L^{-T}L_B^{-T}\mathbf{c}
# \end{equation*}
# and
# \begin{equation*}
# \Sigma = K_{nn} - K_{nu}L^{-T}(I - B^{-1})L^{-1}K_{un}
# \end{equation*}
#
# Where the mean function is not the zero function, the predictive mean should have the mean function evaluated at the test points added to it.

# + [markdown] id="GX1U-fYPKPrt"
# ## What can be cached?
#
# We cache two separate values: $\alpha$ and $Q^{-1}$. These correspond to the parts of the mean and covariance functions respectively which do not depend upon the test points. In the case of the GPR these are the same value:
# \begin{equation*}
#     \alpha = Q^{-1} = [K_{mm} + \sigma^2I]^{-1}
# \end{equation*}
# in the case of the VGP and SVGP model these are:
# \begin{equation*}
#     \alpha = K_{uu}^{-1}\mathbf{u}\\ Q^{-1} = K_{uu}^{-1}
# \end{equation*}
# and in the case of the SGPR model these are:
# \begin{equation*}
#     \alpha = L^{-T}L_B^{-T}\mathbf{c}\\ Q^{-1} = L^{-T}(I - B^{-1})L^{-1}
# \end{equation*}
#
#
# Note that in the (S)VGP case, $\alpha$ is the parameter as proposed by Opper and Archambeau for the mean of the predictive distribution.

# +
import gpflow
import numpy as np

# Create some data
X = np.linspace(-1.1, 1.1, 1000)[:, None]
Y = np.sin(X)
Xnew = np.linspace(-1.1, 1.1, 1000)[:, None]

# + [markdown] id="FzCgor4nKUcW"
#
# ## GPR Example
#
# We will construct a GPR model to demonstrate the faster predictions from using the cached data in the GPFlow posterior classes (subclasses of `gpflow.posteriors.AbstractPosterior`).

# + id="BMnIdXNiKU6t"
model = gpflow.models.GPR(
    (X, Y),
    gpflow.kernels.SquaredExponential(),
)
# -

# The `predict_f` method on the `GPModel` class performs no caching.

# %%timeit
model.predict_f(Xnew)

# To make use of the caching, first retrieve the posterior class from the model. The posterior class has methods to predict the parameters of marginal distributions at test points, in the same way as the `predict_f` method of the `GPModel`.
posterior = model.posterior()

# %%timeit
posterior.predict_f(Xnew)

# ## SVGP Example
#
# Likewise, we will construct an SVGP model to demonstrate the faster predictions from using the cached data in the GPFlow posterior classes.

# + id="BMnIdXNiKU6t"
model = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(),
    gpflow.likelihoods.Gaussian(),
    np.linspace(-1.1, 1.1, 1000)[:, None],
)
# -

# The `predict_f` method on the `GPModel` class performs no caching.

# %%timeit
model.predict_f(Xnew)

# And again using the posterior object and caching

posterior = model.posterior()

# %%timeit
posterior.predict_f(Xnew)

# ## SGPR Example
#
# And finally, we follow the same approach this time for the SGPR case.

model = gpflow.models.SGPR(
    (X, Y), gpflow.kernels.SquaredExponential(), np.linspace(-1.1, 1.1, 1000)[:, None]
)

# The predict_f method on the instance performs no caching.

# %%timeit
model.predict_f(Xnew)

# Using the posterior object instead:

posterior = model.posterior()

# %%timeit
posterior.predict_f(Xnew)
