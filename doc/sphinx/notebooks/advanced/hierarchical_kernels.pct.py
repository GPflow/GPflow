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
# # Hierarchical kernels
#
# A *hierarchical* search space is one in which some input dimensions are only
# meaningful when a corresponding indicator variable takes a particular value.
# Concretely, you might be optimising over a four-dimensional input
# $(x_1, y_1, x_2, x_3)$ where $x_1$ is always present, $y_1 \in \{0, 1\}$ is
# an indicator, $x_2$ is meaningful only when $y_1 = 1$, and $x_3$ is
# meaningful only when $y_1 = 0$.
#
# A useful covariance on such a space must respect the activation structure:
# a point whose conditional feature is *inactive* should not be confused with
# one whose feature is *active and equal in value* — the two live in
# fundamentally different regions of the space.
#
# This notebook walks through GPflow's two hierarchical kernels —
# `ArcHierarchical` (Swersky et al., 2014) and `WedgeHierarchical`
# (Horn et al., 2019) — on a small synthetic disjunctive function.
#
# ## Worked example: fit a GP on a synthetic disjunctive function
#
# $$f(x_1, y_1, x_2, x_3) =
#     \sin(2\pi x_1)
#   \;+\; \mathbf{1}[y_1 = 1] \cdot \tfrac{1}{2} \cos(\pi x_2 / 5)
#   \;+\; \mathbf{1}[y_1 = 0] \cdot \tfrac{1}{2} x_3.$$
#
# The conditional kernel must accurately represent the disjunctive structure.

import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)


def objective(X: np.ndarray) -> np.ndarray:
    x1, y1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (
        np.sin(2.0 * np.pi * x1)
        + (y1 > 0.5).astype(float) * 0.5 * np.cos(np.pi * x2 / 5.0)
        + (y1 < 0.5).astype(float) * 0.5 * x3
    ).reshape(-1, 1)


def sample_inputs(n: int) -> np.ndarray:
    x1 = np.random.uniform(0.0, 1.0, size=n)
    y1 = np.random.randint(0, 2, size=n).astype(float)
    x2 = np.random.uniform(0.0, 5.0, size=n)
    x3 = np.random.uniform(-1.0, 1.0, size=n)
    return np.stack([x1, y1, x2, x3], axis=-1)


X_train = sample_inputs(40)
Y_train = objective(X_train) + 0.05 * np.random.randn(40, 1)
X_test = np.array(
    [
        [0.3, 1.0, 2.0, 0.0],  # y1 = 1 branch
        [0.3, 0.0, 0.0, 0.5],  # y1 = 0 branch
        [0.7, 1.0, 4.0, 0.0],
        [0.7, 0.0, 0.0, -0.4],
    ]
)


# %% [markdown]
# ## Three axioms for a conditional distance
#
# Let $\mathbf{x}^{nc}$ be the always-active (unconditional) coordinates and
# $\mathbf{x}^{c}$ the indicator-gated (conditional) ones. A useful
# per-dimension distance $d_i(\mathbf{x}, \mathbf{x}')$ on a conditional
# dimension $i$ should satisfy:
#
# 1. **both inactive:** $d_i = 0$ — the value is meaningless on either side.
# 2. **both active:** $d_i$ is a function of the difference in feature value.
# 3. **incomparable:** (one active, one inactive) $d_i$ is positive and
#    places the two points in distinct regions of the embedded space.
#
# Both kernels enforce these axioms by embedding each conditional dimension
# into $\mathbb{R}^2$ — inactive points map to the origin, active points to a
# value-dependent point off the origin — and evaluating a stationary base
# kernel on the joint embedded vector.

# %% [markdown]
# ## Describe the search space to the kernel
#
# The kernel takes a single piece of information:
#
# * `hierarchy` — a sequence of `HierarchyNode`s. Each node binds a group of
#   feature columns to the `ActivityCondition` that gates them (and carries
#   the per-feature `(lower, upper)` bounds used to normalise to $[0, 1]$).
#   An empty `ActivityCondition` makes the node's features unconditional.
#   The keys of each `ActivityCondition` are column indices in $X$ (the same
#   coordinate system as `feature_dims`) — the set of indicator columns is
#   derived automatically.
#
# For our four-variable example, the column layout in $X$ is
# $[x_1, y_1, x_2, x_3]$ — so $x_1$ lives in a shared node, $x_2$ in a node
# gated by $y_1 = 1$ (column 1), and $x_3$ in a node gated by $y_1 = 0$:

# %%
from gpflow.kernels import ActivityCondition, HierarchyNode

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

# %% [markdown]
# ## Construct an Arc kernel and inspect it

# %%

from gpflow.kernels import ArcHierarchical, Constant
from gpflow.models import GPR
from gpflow.optimizers import Scipy

arc = ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
print("conditional columns:", arc.n_cond_dims)
print("unconditional columns:", arc.n_uncond_dims)
print("indicator columns (derived):", arc.indicator_dims)
print("angle init:", arc.angle.numpy())
print("radius init:", arc.radius.numpy())

# %% [markdown]
# Wrap in a Constant() factor so the GP can learn an overall variance.
arc_for_fit = ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
kernel = Constant() * arc_for_fit
gpr = GPR(data=(X_train, Y_train), kernel=kernel, noise_variance=0.05)

print(f"LML before fit: {gpr.log_marginal_likelihood().numpy():+.3f}")
Scipy().minimize(
    gpr.training_loss, gpr.trainable_variables, options={"maxiter": 100}
)
print(f"LML after fit:  {gpr.log_marginal_likelihood().numpy():+.3f}")
print("learnt angle :", arc_for_fit.angle.numpy())
print("learnt radius:", arc_for_fit.radius.numpy())

mean, var = gpr.predict_f(X_test)
truth = objective(X_test).ravel()
for x, m, v, t in zip(X_test, mean.numpy().ravel(), var.numpy().ravel(), truth):
    print(
        f"  x = {x.tolist()}  ->  mean = {m:+.3f}  var = {v:.3f}  truth = {t:+.3f}"
    )

# %% [markdown]
# ## A Wedge variant
#
# The Wedge kernel replaces Arc's circle embedding with a triangular one:
#
# $$\phi_c(v_c, m_c) = \big(
#     (\theta_1 v_c + \theta_2 v_c \cos\rho)\, m_c,\;
#     (\theta_2 v_c \sin\rho)\, m_c
#   \big).$$
#
# The "incomparable" distance now scales with the active value $v_c$ rather
# than being constant in it.
# GPR models may be constructed and trained in the same way as the arc above.

# %%
from gpflow.kernels import WedgeHierarchical

wedge = WedgeHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
kernel = Constant() * wedge
print("initial theta1:", wedge.theta1.numpy())
print("initial theta2:", wedge.theta2.numpy())
print("initial rho:   ", wedge.rho.numpy())

# %% [markdown]
# ## Validating composed kernels against the axioms
#
# Once you start composing a hierarchical kernel with other kernels — scaling
# with `Constant()`, adding a `Matern52` on the unconditional dims, summing
# two hierarchical kernels — it stops being obvious whether the result still
# respects the three axioms. GPflow ships `validate_hierarchical_axioms` for
# exactly this: hand it any kernel plus the hierarchy spec, and it
# numerically checks the kernel-level shadows of each axiom.

# %%
from gpflow.kernels import Matern52, validate_hierarchical_axioms

safe = Constant() * ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
report = validate_hierarchical_axioms(safe, hierarchy, seed=0)
print(report)

# %% [markdown]
# Multiplying by a positive `Constant()` just rescales `K` and preserves
# every axiom, so the report is all-`PASS`. Now build a kernel that
# deliberately ignores the hierarchy — add a plain `Matern52` over the full
# input space — and re-run the validator:

# %%
broken = ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4))) + Matern52(
    active_dims=list(range(4))
)
report = validate_hierarchical_axioms(broken, hierarchy, seed=0)
print(report)
assert not report.passed
print("axiom-1 violations:", [c.max_violation for c in report.for_axiom(1)])

# %% [markdown]
# Axiom 1 fails: the `Matern52` term responds to changes in a conditional
# feature value regardless of whether the activity condition is satisfied,
# so `K(x, x)` and `K(x, x')` differ even though both points are inactive
# on that feature. Axioms 2 and 3 still pass — they were never the
# discriminating predicates for this kind of break.
