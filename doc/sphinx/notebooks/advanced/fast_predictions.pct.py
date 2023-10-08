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
# The default behaviour of `predict_f` and `predict_y` in GPflow models is to compute the predictions from scratch on each call. This is convenient when predicting and training are interleaved, and simplifies the use of these models. There are some use cases, such as Bayesian optimisation, where prediction (at different test points) happens much more frequently than training; or, under ths MCMC settings, the sampled posterior samples are re-used many times to generate different predictions. In these cases it is convenient to cache parts of the calculation which do not depend upon the test points, and reuse those parts between predictions.
#
# There are four models to which we want to add this caching capability: GPR, (S)VGP, SGPR and GPMC. The VGP and SVGP can be considered together; the difference between the models is whether to condition on the full training data set (VGP) or on the inducing variables (SVGP). For the case of the Bayesian framework, we will demo how to cache and rebuild models for faster predictions based on the sampled posterior samples.

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
# \alpha = Q^{-1} = [K_{mm} + \sigma^2I]^{-1}
# \end{equation*}
# in the case of the VGP and SVGP model these are:
# \begin{equation*}
# \alpha = K_{uu}^{-1}\mathbf{u}\\ Q^{-1} = K_{uu}^{-1}
# \end{equation*}
# and in the case of the SGPR model these are:
# \begin{equation*}
# \alpha = L^{-T}L_B^{-T}\mathbf{c}\\ Q^{-1} = L^{-T}(I - B^{-1})L^{-1}
# \end{equation*}
#
#
# Note that in the (S)VGP case, $\alpha$ is the parameter as proposed by Opper and Archambeau for the mean of the predictive distribution.

# +
import numpy as np

import gpflow
from gpflow.ci_utils import reduce_in_tests
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float


# Create some data
n_data = reduce_in_tests(1000)
X = np.linspace(-1.1, 1.1, n_data)[:, None]
Y = np.sin(X)
Xnew = np.linspace(-1.1, 1.1, n_data)[:, None]
inducing_points = Xnew

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

# The `predict_y` method on the `GPModel` class performs no caching.
# %%timeit
model.predict_y(Xnew)

# We make use of the retrieved posterior to compute the mean and variance of the held-out data at the input points, in a faster way.
# %%timeit
model.predict_y_faster(Xnew, posteriors=posterior)





# ## SVGP Example
#
# Likewise, we will construct an SVGP model to demonstrate the faster predictions from using the cached data in the GPFlow posterior classes.

# + id="BMnIdXNiKU6t"
model = gpflow.models.SVGP(
    gpflow.kernels.SquaredExponential(),
    gpflow.likelihoods.Gaussian(),
    inducing_points,
)
# -

# The `predict_f` method on the `GPModel` class performs no caching.

# %%timeit
model.predict_f(Xnew)

# And again using the posterior object and caching

posterior = model.posterior()

# %%timeit
posterior.predict_f(Xnew)

# The `predict_y` method on the `GPModel` class performs no caching.

# %%timeit
model.predict_y(Xnew)

# We make use of the retrieved posterior to compute the mean and variance of the held-out data at the input points, in a faster way.

# %%timeit
model.predict_y_faster(Xnew, posteriors=posterior)


# ## SGPR Example
#
# And finally, we follow the same approach this time for the SGPR case.

model = gpflow.models.SGPR(
    (X, Y), gpflow.kernels.SquaredExponential(), inducing_points
)

# The predict_f method on the instance performs no caching.

# %%timeit
model.predict_f(Xnew)

# Using the posterior object instead:

posterior = model.posterior()

# %%timeit
posterior.predict_f(Xnew)

# The `predict_y` method on the `GPModel` class performs no caching.

# %%timeit
model.predict_y(Xnew)

# Making predictions fo y faster in the same manner as the other cases.

# %%timeit
model.predict_y_faster(Xnew, posteriors=posterior)



# ## MCMC for GPR Example
#
# Faster predictions for the sampled hyperparameters in Gaussian process regression
# GPR with HMC

# Model setup and Hyperparameters sampling. See the [MCMC (Markov Chain Monte Carlo)](../advanced/mcmc.ipynb) for more details.
# %%
data = (X, Y) 
kernel = gpflow.kernels.Matern52(lengthscales=0.3)
mean_function = gpflow.mean_functions.Linear(1.0, 0.0)
model = gpflow.models.GPR(data, kernel, mean_function, noise_variance=0.01)
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(model.training_loss, model.trainable_variables)
print(f"log posterior density at optimum: {model.log_posterior_density()}")
model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.mean_function.A.prior = tfd.Normal(f64(0.0), f64(10.0))
model.mean_function.b.prior = tfd.Normal(f64(0.0), f64(10.0))
gpflow.utilities.print_summary(model)

# Sampling hyperparameters
num_burnin_steps = 5 
num_samples = 100

# Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
hmc_helper = gpflow.optimizers.SamplingHelper(
    model.log_posterior_density, model.trainable_parameters
)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
)
@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

samples, traces = run_chain_fn()
parameter_samples = hmc_helper.convert_to_constrained_values(samples)


# standard model predictions
# %%timeit
for _ in range(0, 100):
    i = np.random.randint(0, 100)
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    model.predict_f(Xnew)

# Storing esssential calculation results to make faster predictions later.
# # %%
Cache = {}
for i in range(0, num_samples):
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    Cache[i] = model.posterior().cache

# Retrieving the caching to make faster precitions.
# %%timeit
for _ in range(0, 100):
    i = np.random.randint(0, 100)
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    model.predict_f_loaded_cache(Xnew, Cache[i])

# Now, the faster predictions can be made as long as the variables Cache and samples are stored in the local. 
# %%
# For example, if we delete the old model
del(model)
# build a new model called model2 with identical initial setup as the old model
kernel = gpflow.kernels.Matern52(lengthscales=0.3)
mean_function = gpflow.mean_functions.Linear(1.0, 0.0)
model2 = gpflow.models.GPR(data, kernel, mean_function, noise_variance=0.01)
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(model2.training_loss, model2.trainable_variables)
print(f"log posterior density at optimum: {model2.log_posterior_density()}")
model2.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
model2.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model2.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model2.mean_function.A.prior = tfd.Normal(f64(0.0), f64(10.0))
model2.mean_function.b.prior = tfd.Normal(f64(0.0), f64(10.0))

# %%timeit
# we can still make faster predictions through variables Cache and samples 
for _ in range(0, 100):
    i = np.random.randint(0, 100)
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    model2.predict_f_loaded_cache(Xnew, Cache[i])

# %%timeit
# as well as for faster predictions for predict_y
for _ in range(0, 100):
    i = np.random.randint(0, 100)
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    model2.predict_y_loaded_cache(Xnew, Cache[i])

# ## MCMC for Gaussian process models (GPMC)
# This is similar as the case of GPR. 

# %%
# Create some data
rng = np.random.RandomState(14)
X = np.linspace(-6, 6, 200)
Y = rng.exponential(np.sin(X) ** 2)
data = (X[:, None], Y[:, None])
Xtest = np.linspace(-7, 7, 100)[:, None]

# model setup
kernel = gpflow.kernels.Matern32() + gpflow.kernels.Constant()
likelihood = gpflow.likelihoods.Exponential()
model = gpflow.models.GPMC(data, kernel, likelihood)

model.kernel.kernels[0].lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.kernel.kernels[0].variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.kernel.kernels[1].variance.prior = tfd.Gamma(f64(1.0), f64(1.0))

gpflow.utilities.print_summary(model)

# sampling hyperparameters
optimizer = gpflow.optimizers.Scipy()
maxiter =300 
_ = optimizer.minimize(
    model.training_loss, model.trainable_variables, options=dict(maxiter=maxiter)
)
num_burnin_steps = 10
num_samples = 100

# Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
hmc_helper = gpflow.optimizers.SamplingHelper(
    model.log_posterior_density, model.trainable_parameters
)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
)

adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
)

@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )

samples, _ = run_chain_fn()

# %%timeit
for _ in range(0, 100):
    i = np.random.randint(0, 100)
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    model.predict_f(Xtest)

# %%
# Storing esssential calculation results to make faster predictions later.
Cache = {}
for i in range(0, num_samples):
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    Cache[i] = model.posterior().cache

# %%timeit
for _ in range(0, 100):
    i = np.random.randint(0, 100)
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    model.predict_f_loaded_cache(Xtest, Cache[i])



