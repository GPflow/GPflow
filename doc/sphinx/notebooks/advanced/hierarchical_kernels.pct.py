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
# # Hierarchical (Arc and Wedge) kernels
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

import numpy as np
import tensorflow as tf

# %%
import gpflow
from gpflow.kernels import ActivityCondition, ArcHierarchical, WedgeHierarchical

np.random.seed(1793)
tf.random.set_seed(1793)

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
# The kernel needs four pieces of information:
#
# * `feature_dims` — column indices of real-valued features in the flat input;
# * `feature_bounds` — `[D_f, 2]` `(lower, upper)` per feature, used for
#   normalisation to $[0, 1]$;
# * `indicator_dims` — column indices of the integer-valued indicators;
# * `activity_conditions` — one `ActivityCondition` per feature column,
#   describing the AND-conjunction of indicator requirements that gates it
#   (an empty `ActivityCondition` means the column is unconditional).
#
# For our four-variable example, column layout in $X$ is
# $[x_1, y_1, x_2, x_3]$:

# %%
feature_dims = [0, 2, 3]
feature_bounds = tf.constant([[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=tf.float64)
indicator_dims = [1]
activity_conditions = [
    ActivityCondition(),  # x1: unconditional
    ActivityCondition({0: 1}),  # x2: active when indicator 0 == 1
    ActivityCondition({0: 0}),  # x3: active when indicator 0 == 0
]

# %% [markdown]
# ## Construct an Arc kernel and inspect it

# %%
arc = ArcHierarchical(
    feature_dims=feature_dims,
    feature_bounds=feature_bounds,
    indicator_dims=indicator_dims,
    activity_conditions=activity_conditions,
)
print("conditional columns:", arc._n_cond)
print("unconditional columns:", arc._n_uncond)
print("angle init:", arc.angle.numpy())
print("radius init:", arc.radius.numpy())

# %% [markdown]
# ## Worked example: fit a GP on a synthetic disjunctive function
#
# $$f(x_1, y_1, x_2, x_3) =
#     \sin(2\pi x_1)
#   \;+\; \mathbf{1}[y_1 = 1] \cdot \tfrac{1}{2} \cos(\pi x_2 / 5)
#   \;+\; \mathbf{1}[y_1 = 0] \cdot \tfrac{1}{2} x_3.$$
#
# The conditional kernel must accurately represent the disjunctive structure.


# %%
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

# Wrap in a Constant() factor so the GP can learn an overall variance.
arc_for_fit = ArcHierarchical(
    feature_dims=feature_dims,
    feature_bounds=feature_bounds,
    indicator_dims=indicator_dims,
    activity_conditions=activity_conditions,
)
kernel = gpflow.kernels.Constant() * arc_for_fit
gpr = gpflow.models.GPR(data=(X_train, Y_train), kernel=kernel, noise_variance=0.05)

print(f"LML before fit: {gpr.log_marginal_likelihood().numpy():+.3f}")
gpflow.optimizers.Scipy().minimize(
    gpr.training_loss, gpr.trainable_variables, options={"maxiter": 100}
)
print(f"LML after fit:  {gpr.log_marginal_likelihood().numpy():+.3f}")
print("learnt angle :", arc_for_fit.angle.numpy())
print("learnt radius:", arc_for_fit.radius.numpy())

# %% [markdown]
# ## Predict on held-out points
#
# The kernel places the two branches on different parts of the embedded
# space, so predicted means follow the corresponding branch's signal.

# %%
X_test = np.array(
    [
        [0.3, 1.0, 2.0, 0.0],  # y1 = 1 branch
        [0.3, 0.0, 0.0, 0.5],  # y1 = 0 branch
        [0.7, 1.0, 4.0, 0.0],
        [0.7, 0.0, 0.0, -0.4],
    ]
)
mean, var = gpr.predict_f(X_test)
truth = objective(X_test).ravel()
for x, m, v, t in zip(X_test, mean.numpy().ravel(), var.numpy().ravel(), truth):
    print(f"  x = {x.tolist()}  ->  mean = {m:+.3f}  var = {v:.3f}  truth = {t:+.3f}")

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

# %%
wedge = WedgeHierarchical(
    feature_dims=feature_dims,
    feature_bounds=feature_bounds,
    indicator_dims=indicator_dims,
    activity_conditions=activity_conditions,
)
kernel = gpflow.kernels.Constant() * wedge
gpr = gpflow.models.GPR(data=(X_train, Y_train), kernel=kernel, noise_variance=0.05)
print(f"LML before fit: {gpr.log_marginal_likelihood().numpy():+.3f}")
gpflow.optimizers.Scipy().minimize(
    gpr.training_loss, gpr.trainable_variables, options={"maxiter": 100}
)
print(f"LML after fit:  {gpr.log_marginal_likelihood().numpy():+.3f}")
print("learnt theta1:", wedge.theta1.numpy())
print("learnt theta2:", wedge.theta2.numpy())
print("learnt rho:   ", wedge.rho.numpy())

# %% [markdown]
# ## Predict on held-out points
#
# The kernel places the two branches on different parts of the embedded
# space, so predicted means follow the corresponding branch's signal.

# %%
X_test = np.array(
    [
        [0.3, 1.0, 2.0, 0.0],  # y1 = 1 branch
        [0.3, 0.0, 0.0, 0.5],  # y1 = 0 branch
        [0.7, 1.0, 4.0, 0.0],
        [0.7, 0.0, 0.0, -0.4],
    ]
)
mean, var = gpr.predict_f(X_test)
truth = objective(X_test).ravel()
for x, m, v, t in zip(X_test, mean.numpy().ravel(), var.numpy().ravel(), truth):
    print(f"  x = {x.tolist()}  ->  mean = {m:+.3f}  var = {v:.3f}  truth = {t:+.3f}")
