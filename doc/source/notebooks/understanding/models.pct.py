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
# # Manipulating GPflow models
#
# One of the key ingredients in GPflow is the model class, which enables you to carefully control parameters. This notebook shows how some of these parameter control features work, and how to build your own model with GPflow. First we'll look at:
#
#  - how to view models and parameters
#  - how to set parameter values
#  - how to constrain parameters (for example, variance > 0)
#  - how to fix model parameters
#  - how to apply priors to parameters
#  - how to optimize models
#
# Then we'll show how to build a simple logistic regression model, demonstrating the ease of the parameter framework.
#
# GPy users should feel right at home, but there are some small differences.
#
# First, let's deal with the usual notebook boilerplate and make a simple GP regression model. See [Basic (Gaussian likelihood) GP regression model](../basics/regression.ipynb) for specifics of the model; we just want some parameters to play with.

# %%
import numpy as np
import gpflow
import tensorflow_probability as tfp
from gpflow.utilities import print_summary, set_trainable, to_default_float

# %% [markdown]
# We begin by creating a very simple GP regression model:

# %%
# generate toy data
np.random.seed(1)
X = np.random.rand(20, 1)
Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(20, 1) * 0.01

m = gpflow.models.GPR((X, Y), kernel=gpflow.kernels.Matern32() + gpflow.kernels.Linear())

# %% [markdown]
# ## Viewing, getting, and setting parameters
# You can display the state of the model in a terminal by using `print_summary(m)`. You can change the display format using the `fmt` keyword argument, e.g. `'html'`. In a notebook, you can also use `fmt='notebook'` or set the default printing format as `notebook`:

# %%
print_summary(m, fmt="notebook")

# %%
gpflow.config.set_default_summary_fmt("notebook")

# %% [markdown]
# This model has four parameters. The kernel is made of the sum of two parts. The first (counting from zero) is a Matern32 kernel that has a variance parameter and a lengthscales parameter; the second is a linear kernel that has only a variance parameter. There is also a parameter that controls the variance of the noise, as part of the likelihood.
#
# All the model variables have been initialized at `1.0`. You can access individual parameters in the same way that you display the state of the model in a terminal; for example, to see all the parameters that are part of the likelihood, run:

# %%
print_summary(m.likelihood)

# %% [markdown]
# This gets more useful with more complex models!

# %% [markdown]
# To set the value of a parameter, just use `assign()`:

# %%
m.kernel.kernels[0].lengthscales.assign(0.5)
m.likelihood.variance.assign(0.01)
print_summary(m, fmt="notebook")

# %% [markdown]
# ## Constraints and trainable variables
#
# GPflow helpfully creates an unconstrained representation of all the variables. In the previous example, all the variables are constrained positively (see the **transform** column in the table); the unconstrained representation is given by $\alpha = \log(\exp(\theta)-1)$. The `trainable_parameters` property returns the constrained values:

# %%
m.trainable_parameters

# %% [markdown]
# Each parameter has an `unconstrained_variable` attribute that enables you to access the unconstrained value as a TensorFlow `Variable`.

# %%
p = m.kernel.kernels[0].lengthscales
p.unconstrained_variable

# %% [markdown]
# You can also check the unconstrained value as follows:

# %%
p.transform.inverse(p)

# %% [markdown]
# Constraints are handled by the Bijector classes from the `tensorflow_probability` package. You might prefer to use the constraint $\alpha = \log(\theta)$; this is easily done by replacing the parameter with one that has a different `transform` attribute (here we make sure to copy all other attributes across from the old parameter; this is not necessary when there is no `prior` and the `trainable` state is still the default of `True`):

# %%
old_parameter = m.kernel.kernels[0].lengthscales
new_parameter = gpflow.Parameter(
    old_parameter,
    trainable=old_parameter.trainable,
    prior=old_parameter.prior,
    name=old_parameter.name.split(":")[0],  # tensorflow is weird and adds ':0' to the name
    transform=tfp.bijectors.Exp(),
)
m.kernel.kernels[0].lengthscales = new_parameter

# %% [markdown]
# Though the lengthscale itself remains the same, the unconstrained lengthscale has changed:

# %%
p.transform.inverse(p)

# %% [markdown]
# To replace the `transform` of a parameter you need to recreate the parameter with updated transform:

# %%
p = m.kernel.kernels[0].variance
m.kernel.kernels[0].variance = gpflow.Parameter(p.numpy(), transform=tfp.bijectors.Exp())

# %%
print_summary(m, fmt="notebook")

# %% [markdown]
# ## Changing whether a parameter will be trained in optimization
#
# Another helpful feature is the ability to fix parameters. To do this, simply set the `trainable` attribute to `False`; this is shown in the **trainable** column of the representation, and the corresponding variable is removed from the free state.

# %%
set_trainable(m.kernel.kernels[1].variance, False)
print_summary(m)

# %%
m.trainable_parameters

# %% [markdown]
# To unfix a parameter, just set the `trainable` attribute to `True` again.

# %%
set_trainable(m.kernel.kernels[1].variance, True)
print_summary(m)

# %% [markdown]
# **NOTE:** If you want to recursively change the `trainable` status of an object that *contains* parameters, you **must** use the `set_trainable()` utility function.
#
# A module (e.g. a model, kernel, likelihood, ... instance) does not have a `trainable` attribute:

# %%
try:
    m.kernel.trainable
except AttributeError:
    print(f"{m.kernel.__class__.__name__} does not have a trainable attribute")

# %%
set_trainable(m.kernel, False)
print_summary(m)

# %% [markdown]
# ## Priors
#
# You can set priors in the same way as transforms and trainability, by using `tensorflow_probability` distribution objects. Let's set a Gamma prior on the variance of the Matern32 kernel.

# %%
k = gpflow.kernels.Matern32()
k.variance.prior = tfp.distributions.Gamma(to_default_float(2), to_default_float(3))

print_summary(k)

# %%
m.kernel.kernels[0].variance.prior = tfp.distributions.Gamma(
    to_default_float(2), to_default_float(3)
)
print_summary(m)


# %% [markdown]
# ## Optimization
#
# To optimize your model, first create an instance of an optimizer (in this case, `gpflow.optimizers.Scipy`), which has optional arguments that are passed to `scipy.optimize.minimize` (we minimize the negative log likelihood). Then, call the `minimize` method of that optimizer, with your model as the optimization target. Variables that have priors are maximum a priori (MAP) estimated, that is, we add the log prior to the log likelihood, and otherwise use Maximum Likelihood.

# %%
opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, variables=m.trainable_variables)

# %% [markdown]
# ## Building new models
#
# To build new models, you'll need to inherit from `gpflow.models.BayesianModel`.
# Parameters are instantiated with `gpflow.Parameter`.
# You might also be interested in `gpflow.Module` (a subclass of `tf.Module`), which acts as a 'container' for `Parameter`s (for example, kernels are `gpflow.Module`s).
#
# In this very simple demo, we'll implement linear multiclass classification.
#
# There are two parameters: a weight matrix and a bias (offset). You can use
# Parameter objects directly, like any TensorFlow tensor.
#
# The training objective depends on the type of model; it may be possible to
# implement the exact (log)marginal likelihood, or only a lower bound to the
# log marginal likelihood (ELBO). You need to implement this as the
# `maximum_log_likelihood_objective` method. The `BayesianModel` parent class
# provides a `log_posterior_density` method that returns the
# `maximum_log_likelihood_objective` plus the sum of the log-density of any priors
# on hyperparameters, which can be used for MCMC.
# GPflow provides mixin classes that define a `training_loss` method
# that returns the negative of (maximum likelihood objective + log prior
# density) for MLE/MAP estimation to be passed to optimizer's `minimize`
# method. Models that derive from `InternalDataTrainingLossMixin` are expected to store the data internally, and their `training_loss` does not take any arguments and can be passed directly to `minimize`.
# Models that take data as an argument to their `maximum_log_likelihood_objective` method derive from `ExternalDataTrainingLossMixin`, which provides a `training_loss_closure` to take the data and return the appropriate closure for `optimizer.minimize`.
# This is also discussed in the [GPflow with TensorFlow 2 notebook](../intro_to_gpflow2.ipynb).

# %%
import tensorflow as tf


class LinearMulticlass(gpflow.models.BayesianModel, gpflow.models.InternalDataTrainingLossMixin):
    # The InternalDataTrainingLossMixin provides the training_loss method.
    # (There is also an ExternalDataTrainingLossMixin for models that do not encapsulate data.)

    def __init__(self, X, Y, name=None):
        super().__init__(name=name)  # always call the parent constructor

        self.X = X.copy()  # X is a NumPy array of inputs
        self.Y = Y.copy()  # Y is a 1-of-k (one-hot) representation of the labels

        self.num_data, self.input_dim = X.shape
        _, self.num_classes = Y.shape

        # make some parameters
        self.W = gpflow.Parameter(np.random.randn(self.input_dim, self.num_classes))
        self.b = gpflow.Parameter(np.random.randn(self.num_classes))

        # ^^ You must make the parameters attributes of the class for
        # them to be picked up by the model. i.e. this won't work:
        #
        # W = gpflow.Parameter(...    <-- must be self.W

    def maximum_log_likelihood_objective(self):
        p = tf.nn.softmax(
            tf.matmul(self.X, self.W) + self.b
        )  # Parameters can be used like a tf.Tensor
        return tf.reduce_sum(tf.math.log(p) * self.Y)  # be sure to return a scalar


# %% [markdown]
# ...and that's it. Let's build a really simple demo to show that it works.

# %%
np.random.seed(123)
X = np.vstack(
    [
        np.random.randn(10, 2) + [2, 2],
        np.random.randn(10, 2) + [-2, 2],
        np.random.randn(10, 2) + [2, -2],
    ]
)
Y = np.repeat(np.eye(3), 10, 0)

import matplotlib.pyplot as plt

plt.style.use("ggplot")
# %matplotlib inline

plt.rcParams["figure.figsize"] = (12, 6)
_ = plt.scatter(X[:, 0], X[:, 1], 100, np.argmax(Y, 1), lw=2, cmap=plt.cm.viridis)

# %%
m = LinearMulticlass(X, Y)
m


# %%
opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, variables=m.trainable_variables)

# %%
xx, yy = np.mgrid[-4:4:200j, -4:4:200j]
X_test = np.vstack([xx.flatten(), yy.flatten()]).T
f_test = np.dot(X_test, m.W.numpy()) + m.b.numpy()
p_test = np.exp(f_test)
p_test /= p_test.sum(1)[:, None]

# %%
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.contour(xx, yy, p_test[:, i].reshape(200, 200), [0.5], colors="k", linewidths=1)
_ = plt.scatter(X[:, 0], X[:, 1], 100, np.argmax(Y, 1), lw=2, cmap=plt.cm.viridis)

# %% [markdown]
# That concludes the new model example and this notebook. You might want to see for yourself that the `LinearMulticlass` model and its parameters have all the functionality demonstrated here. You could also add some priors and run Hamiltonian Monte Carlo using the HMC optimizer `gpflow.train.HMC` and its `sample` method. See [Markov Chain Monte Carlo (MCMC)](../advanced/mcmc.ipynb) for more information on running the sampler.
