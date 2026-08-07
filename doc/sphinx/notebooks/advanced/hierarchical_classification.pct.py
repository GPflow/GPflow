# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# remove-cell
# pylint: disable=line-too-long,redefined-outer-name

# %% [markdown]
# # Classification on a hierarchical search space
#
# A *hierarchical* (or conditional, or disjunctive) search space is one in which
# some input dimensions are only meaningful when a corresponding indicator
# variable takes a particular value. The
# [hierarchical kernels](hierarchical_kernels.ipynb) notebook introduces
# `ArcHierarchical` and `WedgeHierarchical`, the three axioms a conditional
# covariance must satisfy, and fits a `GPR` to a synthetic regression target.
#
# This notebook is about the *plumbing*: how to wire one of those kernels into a
# **classifier**. Hierarchical kernels are plain `gpflow.kernels.Kernel`
# subclasses with no likelihood-specific code, so they drop straight into `VGP`
# and `SVGP` with a `Bernoulli` likelihood. There are, however, four things that
# will bite you if you do not know about them:
#
# 1. the strict contract between `active_dims` and the hierarchy,
# 2. indicator columns must hold integer-valued floats,
# 3. which kernel hyperparameters are trainable, and which are deliberately not,
# 4. inducing points live in the *full* input space, indicator columns included.
#
# We work through each in turn.

# %% [markdown]
# As usual we start with our imports:

# %%
# hide: begin
import os
import warnings

warnings.simplefilter("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# hide: end

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import norm

import gpflow
from gpflow.ci_utils import reduce_in_tests
from gpflow.kernels import (
    ActivityCondition,
    ArcHierarchical,
    HierarchyNode,
    validate_hierarchical_axioms,
)
from gpflow.utilities import print_summary

# hide: begin
# %matplotlib inline
plt.rcParams["figure.figsize"] = (12, 6)
# hide: end

rng = np.random.default_rng(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## A disjunctive classification problem
#
# We reuse the search space from the
# [hierarchical kernels](hierarchical_kernels.ipynb) notebook. An input row is
# $(x_1, y_1, x_2, x_3)$ where:
#
# * $x_1 \in [0, 1]$ is always active,
# * $y_1 \in \{0, 1\}$ is an indicator,
# * $x_2 \in [0, 5]$ is active only when $y_1 = 1$ (call this *branch A*),
# * $x_3 \in [-1, 1]$ is active only when $y_1 = 0$ (*branch B*).
#
# The latent function is disjunctive: the branch that is switched off
# contributes nothing.
#
# $$f(x_1, y_1, x_2, x_3) =
#     \sin(2\pi x_1)
#   \;+\; \mathbf{1}[y_1 = 1] \cos(\tfrac{2}{5}\pi x_2)
#   \;+\; \mathbf{1}[y_1 = 0] \cdot \tfrac{3}{2} x_3.$$
#
# To turn this into a classification problem we squash $f$ through a standard
# normal CDF and draw a Bernoulli label — exactly the generative model that
# [gpflow.likelihoods.Bernoulli](../../api/gpflow/likelihoods/index.rst#gpflow-likelihoods-bernoulli)
# assumes by default, since its default `invlink` is the inverse probit.
#
# Note how `y_1` is generated: `rng.integers(0, 2).astype(float)`. **Indicator
# columns must hold integer-valued floats.** The kernel rounds them internally
# to decide which features are active, so a value of `0.4` silently becomes
# `0`, and a "soft" or one-hot-scaled indicator will not do what you expect.

# %%
LATENT_SCALE = 2.0


def latent_f(X: np.ndarray) -> np.ndarray:
    x1, y1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (
        np.sin(2.0 * np.pi * x1)
        + (y1 > 0.5) * np.cos(0.4 * np.pi * x2)
        + (y1 < 0.5) * 1.5 * x3
    )


def sample_data(n: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack(
        [
            rng.uniform(0.0, 1.0, size=n),
            rng.integers(0, 2, size=n).astype(float),  # indicator column
            rng.uniform(0.0, 5.0, size=n),
            rng.uniform(-1.0, 1.0, size=n),
        ],
        axis=-1,
    )
    probability = norm.cdf(LATENT_SCALE * latent_f(X))
    Y = (rng.uniform(size=n) < probability).astype(float).reshape(-1, 1)
    return X, Y


X, Y = sample_data(120)
print("X shape:", X.shape, " Y shape:", Y.shape)
print("class balance:", Y.mean())
print("branch A rows:", int((X[:, 1] == 1.0).sum()))
print("branch B rows:", int((X[:, 1] == 0.0).sum()))

# %% [markdown]
# ## Describing the space to the kernel
#
# A `HierarchyNode` binds a group of feature columns to the `ActivityCondition`
# that gates them, plus the `(lower, upper)` bounds used to normalise those
# features to $[0, 1]$. An empty `ActivityCondition` makes the node
# unconditional. The keys of an `ActivityCondition` are *indicator* column
# indices; the set of indicator columns is derived automatically from them, so
# there is no separate `indicator_dims` argument.

# %%
hierarchy = [
    HierarchyNode("shared", feature_dims=[0], feature_bounds=[[0.0, 1.0]]),
    HierarchyNode(
        "branch_A",
        feature_dims=[2],
        feature_bounds=[[0.0, 5.0]],
        activity_condition=ActivityCondition({1: 1}),
    ),
    HierarchyNode(
        "branch_B",
        feature_dims=[3],
        feature_bounds=[[-1.0, 1.0]],
        activity_condition=ActivityCondition({1: 0}),
    ),
]

kernel = ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
print("unconditional feature columns:", kernel.n_uncond_dims)
print("conditional feature columns:  ", kernel.n_cond_dims)
print("indicator columns (derived):  ", kernel.indicator_dims)

# %% [markdown]
# ### The `active_dims` contract
#
# `active_dims` is a *required* keyword argument, and it is checked strictly at
# construction: after `active_dims` slices the input, the feature columns and
# the derived indicator columns must cover $0, \dots, D-1$ **exactly** — no
# gaps, no spare columns. Every index you write in `feature_dims` and in an
# `ActivityCondition` is an index *after* that slicing.
#
# This is deliberately unforgiving: a hierarchical kernel that silently ignored
# a column, or that was handed a column it had no node for, would quietly
# violate the axioms. Getting it wrong raises immediately:

# %%
try:
    ArcHierarchical(hierarchy=hierarchy, active_dims=[0, 1, 2])
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# If your hierarchical block sits inside a wider input — say columns 2 to 5 of a
# six-column $X$ — then pass `active_dims=[2, 3, 4, 5]` and number
# `feature_dims` / `ActivityCondition` keys as `0, 1, 2, 3`.

# %% [markdown]
# ## A VGP classifier
#
# With the kernel built, the model is entirely ordinary. `VGP` keeps a full
# variational posterior over the latent function values at the training inputs,
# which is the simplest thing that works for a non-Gaussian likelihood, and is
# fine at this sample size.

# %%
model = gpflow.models.VGP(
    (X, Y), kernel=kernel, likelihood=gpflow.likelihoods.Bernoulli()
)

print(f"ELBO before training: {model.elbo().numpy():+.3f}")
gpflow.optimizers.Scipy().minimize(
    model.training_loss,
    model.trainable_variables,
    options=dict(maxiter=reduce_in_tests(200)),
)
print(f"ELBO after training:  {model.elbo().numpy():+.3f}")

# %% [markdown]
# ### Which hyperparameters are trainable?

# %%
print_summary(model.kernel, "notebook")

# %% [markdown]
# Three things in that table are worth explaining.
#
# **`base_kernel.lengthscales` is frozen at 1 and not trainable.** The
# hierarchical kernel works by *embedding* the input — normalising each feature
# to $[0, 1]$, then mapping each conditional feature into $\mathbb{R}^2$ — and
# evaluating a stationary base kernel on the embedded vector. The embedding
# already carries a per-dimension scale (`uncond_lengthscales` for the shared
# dimensions, `radius` for the conditional ones), so a base lengthscale would be
# redundant and the pair would be unidentifiable. It is pinned so that it cannot
# fight with the embedding parameters.
#
# **The signal variance is `base_kernel.variance`**, and it *is* trainable. You
# do not need to wrap the kernel in a `Constant()` factor to give the model an
# overall scale — doing so just adds a second, unidentifiable variance.
#
# **`angle` and `radius` are per-conditional-dimension.** They are the geometry
# of the arc embedding: `radius` sets how far an *active* point sits from the
# origin (and therefore how dissimilar active and inactive points are), while
# `angle` sets how much of a circular arc the feature range sweeps out (and
# therefore how quickly two active points decorrelate as their values diverge).
# Inactive points always map to the origin.

# %% [markdown]
# ## What the classifier learned, branch by branch
#
# Because the two branches are gated by different values of the same indicator,
# no single 2-D slice shows the whole model. We plot one panel per branch: for
# branch A we vary $(x_1, x_2)$ with $y_1 = 1$, and for branch B we vary
# $(x_1, x_3)$ with $y_1 = 0$. In each case the *other* branch's column is
# inactive, so we can put anything there.


# %%
def branch_grid(
    indicator: float, feature_col: int, lo: float, hi: float, n: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a grid over (x1, feature_col) at a fixed indicator value."""
    a, b = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(lo, hi, n))
    grid = np.zeros((a.size, 4))
    grid[:, 0] = a.ravel()
    grid[:, 1] = indicator
    grid[:, feature_col] = b.ravel()
    return a, b, grid


def plot_branch(
    ax: plt.Axes,
    indicator: float,
    feature_col: int,
    lo: float,
    hi: float,
    title: str,
) -> None:
    a, b, grid = branch_grid(indicator, feature_col, lo, hi)
    p = model.predict_y(grid)[0].numpy().reshape(a.shape)
    contours = ax.contourf(a, b, p, levels=np.linspace(0, 1, 21), cmap="RdBu")
    ax.contour(a, b, p, levels=[0.5], colors="k", linewidths=2)
    rows = X[:, 1] == indicator
    ax.scatter(
        X[rows, 0],
        X[rows, feature_col],
        c=Y[rows, 0],
        cmap="RdBu",
        edgecolors="k",
        s=45,
    )
    ax.set_xlabel("$x_1$ (shared)")
    ax.set_ylabel(f"$x_{feature_col}$")
    ax.set_title(title)
    plt.colorbar(contours, ax=ax, label="$p(y = 1)$")


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
plot_branch(ax_a, 1.0, 2, 0.0, 5.0, "Branch A: $y_1 = 1$, $x_2$ active")
plot_branch(ax_b, 0.0, 3, -1.0, 1.0, "Branch B: $y_1 = 0$, $x_3$ active")
_ = fig.tight_layout()

# %% [markdown]
# The two panels share the same $\sin(2\pi x_1)$ ridge along the horizontal
# axis — that is the unconditional dimension, learned from all the data — but
# their vertical structure is completely different: oscillatory in $x_2$ on
# branch A, monotone in $x_3$ on branch B. Each branch's data informs only its
# own conditional dimension, which is the whole point of the hierarchy.
#
# We claimed above that the inactive column can hold anything. That is the
# axioms at work, and it is worth verifying rather than believing:

# %%
X_probe = np.array([[0.3, 1.0, 2.0, 0.0]])  # branch A: x3 is inactive
X_perturbed = X_probe.copy()
X_perturbed[:, 3] = -0.9  # scribble over the inactive column

p_probe = model.predict_y(X_probe)[0].numpy()
p_perturbed = model.predict_y(X_perturbed)[0].numpy()
print(f"p(y=1) with x3 =  0.0: {p_probe.item():.10f}")
print(f"p(y=1) with x3 = -0.9: {p_perturbed.item():.10f}")
np.testing.assert_allclose(p_probe, p_perturbed)

# %% [markdown]
# ## Scaling up: SVGP
#
# `VGP` carries one variational parameter per data point, so for larger datasets
# we switch to `SVGP` and a set of $M$ inducing points. Everything about the
# kernel stays the same; the one new question is what an inducing point *is* in
# a hierarchical space.
#
# Inducing points live in the **full, unsliced input space** — all four columns,
# indicator included. That rules out the usual habit of laying them out on a
# `np.linspace` grid, which would put fractional values in the indicator column.
# The simplest correct choice is to sample rows of the training data, which are
# valid points in the space by construction.

# %%
X_big, Y_big = sample_data(2000)

kernel_svgp = ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
Z = X_big[rng.choice(len(X_big), size=40, replace=False)].copy()
svgp = gpflow.models.SVGP(
    kernel_svgp,
    gpflow.likelihoods.Bernoulli(),
    inducing_variable=Z,
    num_data=len(X_big),
)

Z_before = svgp.inducing_variable.Z.numpy().copy()
loss_fn = svgp.training_loss_closure((X_big, Y_big))
optimiser = tf.optimizers.Adam(0.05)


@tf.function
def training_step() -> None:
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, svgp.trainable_variables)
    optimiser.apply_gradients(zip(grads, svgp.trainable_variables))


print(f"ELBO before training: {svgp.elbo((X_big, Y_big)).numpy():+.3f}")
for _ in range(reduce_in_tests(500)):
    training_step()
print(f"ELBO after training:  {svgp.elbo((X_big, Y_big)).numpy():+.3f}")

p_train = svgp.predict_y(X_big)[0].numpy().ravel()
accuracy = ((p_train > 0.5) == (Y_big.ravel() > 0.5)).mean()
print(f"training accuracy:    {accuracy:.3f}")

# %% [markdown]
# ### Inducing points stay valid on their own
#
# `Z` is trainable by default, which raises an obvious worry: if the optimiser
# moves the inducing points, will it drag the indicator column off `0.0`/`1.0`
# and into meaningless territory?
#
# It will not, and the reason is structural. The kernel decides activity by
# *rounding* the indicator columns, and rounding has zero derivative everywhere
# it is defined. So the indicator entries of `Z` receive exactly zero gradient
# and never move, while the feature entries optimise freely. Let us confirm
# both halves of that claim on the trained model:

# %%
Z_var = svgp.inducing_variable.Z.unconstrained_variable
with tf.GradientTape() as tape:
    loss = loss_fn()
Z_grad = tape.gradient(loss, Z_var).numpy()

feature_cols = [0, 2, 3]
print(f"max |dELBO/dZ| , indicator column: {np.abs(Z_grad[:, 1]).max():.3e}")
print(
    f"max |dELBO/dZ| , feature columns : "
    f"{np.abs(Z_grad[:, feature_cols]).max():.3e}"
)

Z_after = svgp.inducing_variable.Z.numpy()
moved_ind = np.abs(Z_after[:, 1] - Z_before[:, 1]).max()
moved_feat = np.abs(Z_after[:, feature_cols] - Z_before[:, feature_cols]).max()
print(f"largest move during training, indicator column: {moved_ind:.3e}")
print(f"largest move during training, feature columns : {moved_feat:.3e}")

# %% [markdown]
# So you do not need to `set_trainable(svgp.inducing_variable, False)`, nor
# split `Z` into trainable and fixed parts: the indicator columns are frozen by
# construction and every inducing point remains a legal point in the
# hierarchical space.
#
# The flip side is that whichever branch an inducing point is assigned to at
# initialisation, it keeps for the whole of training — gradient descent cannot
# migrate a point from branch A to branch B. Make sure your initial `Z` covers
# every branch in roughly the proportion you care about. Sampling from the
# training data does this for you when the branches are balanced; when they are
# not, consider stratifying the sample.
#
# One thing that is *not* a problem: the feature columns of `Z` are free to
# wander outside the `feature_bounds` you declared. Those bounds only set the
# affine map into $[0, 1]$ — they are not a constraint, and nothing clips to
# them — so an inducing point slightly outside the data range is harmless.

# %% [markdown]
# ## Checking the axioms of the model's kernel
#
# The likelihood has no bearing on the three axioms — they are a property of the
# covariance alone — but as soon as you *compose* the hierarchical kernel with
# anything else it stops being obvious whether the result still respects them.
# `validate_hierarchical_axioms` takes any kernel plus the hierarchy spec and
# checks the kernel-level shadow of each axiom numerically. Run it on the kernel
# you actually handed to the model:

# %%
report = validate_hierarchical_axioms(svgp.kernel, hierarchy, seed=0)
print(report)
assert report.passed

# %% [markdown]
# The [hierarchical kernels](hierarchical_kernels.ipynb) notebook shows what a
# *failing* report looks like — adding a plain `Matern52` over the full input
# space is the classic way to break axiom 1 — and covers the `WedgeHierarchical`
# alternative to the arc embedding. Everything in this notebook applies to
# `WedgeHierarchical` unchanged; only the names of the embedding parameters
# differ.
