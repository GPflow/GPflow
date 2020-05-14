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
# # Comparing FITC approximation to VFE approximation
# This notebook examines why we prefer the Variational Free Energy (VFE) objective to the Fully Independent Training Conditional (FITC) approximation for our sparse approximations.
#

# %%
import gpflow
import tensorflow as tf
from gpflow.ci_utils import ci_niter  # to speed up automated testing of this notebook
import matplotlib.pyplot as plt

# %matplotlib inline

from FITCvsVFE import (
    getTrainingTestData,
    printModelParameters,
    plotPredictions,
    repeatMinimization,
    stretch,
    plotComparisonFigure,
)

import logging

# logging.disable(logging.WARN)  # do not clutter up the notebook with optimization warnings

# %% [markdown]
# First, we load the training data and plot it together with the exact GP solution (using the `GPR` model):

# %%
# Load the training data:
Xtrain, Ytrain, Xtest, Ytest = getTrainingTestData()


def getKernel():
    return gpflow.kernels.SquaredExponential()


# Run exact inference on training data:
exact_model = gpflow.models.GPR((Xtrain, Ytrain), kernel=getKernel())

opt = gpflow.optimizers.Scipy()
opt.minimize(
    exact_model.training_loss,
    exact_model.trainable_variables,
    method="L-BFGS-B",
    options=dict(maxiter=ci_niter(20000)),
    tol=1e-11,
)

print("Exact model parameters:")
printModelParameters(exact_model)

figA, ax = plt.subplots(1, 1)
ax.plot(Xtrain, Ytrain, "ro")
plotPredictions(ax, exact_model, color="g")


# %%
def initializeHyperparametersFromExactSolution(sparse_model):
    sparse_model.likelihood.variance.assign(exact_model.likelihood.variance)
    sparse_model.kernel.variance.assign(exact_model.kernel.variance)
    sparse_model.kernel.lengthscales.assign(exact_model.kernel.lengthscales)


# %% [markdown]
# We now construct two sparse model using the VFE (`SGPR` model) and FITC (`GPRFITC` model) optimization objectives, with the inducing points being initialized on top of the training inputs, and the model hyperparameters (kernel variance and lengthscales, and likelihood variance) being initialized to the values obtained in the optimization of the exact `GPR` model:

# %%
# Train VFE model initialized from the perfect solution.
VFEmodel = gpflow.models.SGPR((Xtrain, Ytrain), kernel=getKernel(), inducing_variable=Xtrain.copy())

initializeHyperparametersFromExactSolution(VFEmodel)

VFEcb = repeatMinimization(VFEmodel, Xtest, Ytest)  # optimize with several restarts
print("Sparse model parameters after VFE optimization:")
printModelParameters(VFEmodel)

# %%
# Train FITC model initialized from the perfect solution.
FITCmodel = gpflow.models.GPRFITC(
    (Xtrain, Ytrain), kernel=getKernel(), inducing_variable=Xtrain.copy()
)

initializeHyperparametersFromExactSolution(FITCmodel)

FITCcb = repeatMinimization(FITCmodel, Xtest, Ytest)  # optimize with several restarts
print("Sparse model parameters after FITC optimization:")
printModelParameters(FITCmodel)

# %% [markdown]
# Plotting a comparison of the two algorithms, we see that VFE stays at the optimum of exact GPR, whereas the FITC approximation eventually ends up with several inducing points on top of each other, and a worse fit:

# %%
figB, axes = plt.subplots(3, 2, figsize=(20, 16))

# VFE optimization finishes after 10 iterations, so we stretch out the training and test
# log-likelihood traces to make them comparable against FITC:
VFEiters = FITCcb.n_iters
VFElog_likelihoods = stretch(len(VFEiters), VFEcb.log_likelihoods)
VFEhold_out_likelihood = stretch(len(VFEiters), VFEcb.hold_out_likelihood)

axes[0, 0].set_title("VFE", loc="center", fontdict={"fontsize": 22})
plotComparisonFigure(
    Xtrain,
    VFEmodel,
    exact_model,
    axes[0, 0],
    axes[1, 0],
    axes[2, 0],
    VFEiters,
    VFElog_likelihoods,
    VFEhold_out_likelihood,
)

axes[0, 1].set_title("FITC", loc="center", fontdict={"fontsize": 22})
plotComparisonFigure(
    Xtrain,
    FITCmodel,
    exact_model,
    axes[0, 1],
    axes[1, 1],
    axes[2, 1],
    FITCcb.n_iters,
    FITCcb.log_likelihoods,
    FITCcb.hold_out_likelihood,
)

# %% [markdown]
# A more detailed discussion of the comparison between these sparse approximations can be found in [Understanding Probabilistic Sparse Gaussian Process Approximations](http://papers.nips.cc/paper/6477-understanding-probabilistic-sparse-gaussian-process-approximations) by Bauer, van der Wilk, and Rasmussen (2017).
