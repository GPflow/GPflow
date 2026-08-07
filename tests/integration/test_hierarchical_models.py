# Copyright 2026 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration tests for the hierarchical kernels composed with GPflow models.

``tests/gpflow/kernels/test_hierarchical.py`` covers the kernels in isolation.
This module covers what happens once one is handed to a model: that the whole
kernel -> conditional -> posterior -> likelihood path works, and that the two
properties a user of a hierarchical space relies on survive it, namely

* an *inactive* feature column cannot influence a prediction, and
* an inducing point's indicator columns stay integer-valued under training.
"""

from typing import Callable, Tuple, Type

import numpy as np
import pytest
import tensorflow as tf
from check_shapes import ShapeChecker

import gpflow
from gpflow.base import AnyNDArray, RegressionData
from gpflow.keras import tf_keras
from gpflow.kernels import (
    ActivityCondition,
    ArcHierarchical,
    HierarchicalEmbeddingKernel,
    HierarchyNode,
    WedgeHierarchical,
)
from gpflow.likelihoods import Bernoulli
from gpflow.models import (
    SVGP,
    VGP,
    GPModel,
    maximum_log_likelihood_objective,
    training_loss_closure,
)

# Column layout of X: [x1, y1, x2, x3].  x1 is always active; y1 is the
# indicator; x2 is active only when y1 == 1; x3 only when y1 == 0.
SHARED_COL = 0
INDICATOR_COL = 1
BRANCH_A_COL = 2
BRANCH_B_COL = 3
FEATURE_COLS = [SHARED_COL, BRANCH_A_COL, BRANCH_B_COL]

MAXITER = 5
NUM_INDUCING = 8
ADAM_STEPS = 20


class Datum:
    cs = ShapeChecker().check_shape

    rng = np.random.default_rng(20260807)
    n = 40
    X: AnyNDArray = cs(
        np.stack(
            [
                rng.uniform(0.0, 1.0, size=n),
                # The indicator must be integer-valued: the kernel rounds this
                # column to decide which features are active.
                rng.integers(0, 2, size=n).astype(float),
                rng.uniform(0.0, 5.0, size=n),
                rng.uniform(-1.0, 1.0, size=n),
            ],
            axis=-1,
        ),
        "[N, 4]",
    )
    latent = cs(
        np.sin(2.0 * np.pi * X[:, SHARED_COL])
        + (X[:, INDICATOR_COL] > 0.5) * np.cos(0.4 * np.pi * X[:, BRANCH_A_COL])
        + (X[:, INDICATOR_COL] < 0.5) * 1.5 * X[:, BRANCH_B_COL],
        "[N]",
    )
    Y: AnyNDArray = cs((latent > 0.0).astype(float).reshape(-1, 1), "[N, 1]")
    data = X, Y

    # Both branches must be populated, or the unused branch's `angle` / `radius`
    # entry never receives a gradient.
    assert (X[:, INDICATOR_COL] == 1.0).any()
    assert (X[:, INDICATOR_COL] == 0.0).any()


def create_hierarchy() -> Tuple[HierarchyNode, ...]:
    return (
        HierarchyNode("shared", feature_dims=[SHARED_COL], feature_bounds=[[0.0, 1.0]]),
        HierarchyNode(
            "branch_A",
            feature_dims=[BRANCH_A_COL],
            feature_bounds=[[0.0, 5.0]],
            activity_condition=ActivityCondition({INDICATOR_COL: 1}),
        ),
        HierarchyNode(
            "branch_B",
            feature_dims=[BRANCH_B_COL],
            feature_bounds=[[-1.0, 1.0]],
            activity_condition=ActivityCondition({INDICATOR_COL: 0}),
        ),
    )


def create_kernel(
    kernel_class: Type[HierarchicalEmbeddingKernel],
) -> HierarchicalEmbeddingKernel:
    return kernel_class(hierarchy=create_hierarchy(), active_dims=list(range(4)))


def create_inducing_points() -> AnyNDArray:
    # Inducing points live in the full, unsliced input space, indicator column
    # included, so they are sampled from the data rather than laid out on a grid.
    rows = Datum.rng.choice(Datum.n, size=NUM_INDUCING, replace=False)
    Z: AnyNDArray = Datum.X[rows].copy()
    return Z


def vgp(kernel: HierarchicalEmbeddingKernel, data: RegressionData) -> GPModel:
    return VGP(data, kernel=kernel, likelihood=Bernoulli())


def svgp(kernel: HierarchicalEmbeddingKernel, data: RegressionData) -> GPModel:
    return SVGP(
        kernel=kernel,
        likelihood=Bernoulli(),
        inducing_variable=create_inducing_points(),
        num_data=len(data[0]),
    )


CreateModel = Callable[[HierarchicalEmbeddingKernel, RegressionData], GPModel]

KERNEL_CLASSES = (ArcHierarchical, WedgeHierarchical)
CREATE_MODELS = (vgp, svgp)


def _train(model: GPModel) -> None:
    gpflow.optimizers.Scipy().minimize(
        training_loss_closure(model, Datum.data, compile=True),
        variables=model.trainable_variables,
        options=dict(maxiter=MAXITER),
    )


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
@pytest.mark.parametrize("create_model", CREATE_MODELS)
def test_objective_improves(
    kernel_class: Type[HierarchicalEmbeddingKernel], create_model: CreateModel
) -> None:
    model = create_model(create_kernel(kernel_class), Datum.data)

    before = maximum_log_likelihood_objective(model, Datum.data).numpy()
    _train(model)
    after = maximum_log_likelihood_objective(model, Datum.data).numpy()

    assert after > before


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
@pytest.mark.parametrize("create_model", CREATE_MODELS)
def test_predict_y_is_a_valid_probability(
    kernel_class: Type[HierarchicalEmbeddingKernel], create_model: CreateModel
) -> None:
    model = create_model(create_kernel(kernel_class), Datum.data)
    _train(model)

    p, _ = model.predict_y(Datum.X)
    p_np = p.numpy()

    assert np.all(np.isfinite(p_np))
    assert np.all(p_np > 0.0)
    assert np.all(p_np < 1.0)


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
@pytest.mark.parametrize("create_model", CREATE_MODELS)
def test_inactive_feature_column_does_not_affect_predictions(
    kernel_class: Type[HierarchicalEmbeddingKernel], create_model: CreateModel
) -> None:
    """A branch_B value is meaningless when y1 == 1, and must be ignored exactly.

    The embedding multiplies the conditional coordinates by a hard 0/1 activity
    mask, so the inactive contribution is exactly zero and predictions are
    bit-identical.  Asserting exact equality is what catches a regression to a
    soft or straight-through mask.
    """
    model = create_model(create_kernel(kernel_class), Datum.data)
    _train(model)

    branch_a_row = int(np.flatnonzero(Datum.X[:, INDICATOR_COL] == 1.0)[0])
    X_probe = Datum.X[branch_a_row : branch_a_row + 1].copy()
    X_perturbed = X_probe.copy()
    X_perturbed[:, BRANCH_B_COL] += 0.77

    mean, var = model.predict_y(X_probe)
    perturbed_mean, perturbed_var = model.predict_y(X_perturbed)

    np.testing.assert_array_equal(mean.numpy(), perturbed_mean.numpy())
    np.testing.assert_array_equal(var.numpy(), perturbed_var.numpy())


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
@pytest.mark.parametrize("create_model", CREATE_MODELS)
def test_base_kernel_lengthscales_stay_frozen_after_training(
    kernel_class: Type[HierarchicalEmbeddingKernel], create_model: CreateModel
) -> None:
    # The embedding already carries a per-dimension scale, so a base lengthscale
    # would be unidentifiable; it is pinned at construction and must stay pinned.
    kernel = create_kernel(kernel_class)
    model = create_model(kernel, Datum.data)
    _train(model)

    np.testing.assert_allclose(kernel.base_kernel.lengthscales.numpy(), 1.0)
    assert not kernel.base_kernel.lengthscales.trainable


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
def test_inducing_indicator_columns_receive_no_gradient(
    kernel_class: Type[HierarchicalEmbeddingKernel],
) -> None:
    """`Z`'s indicator columns are frozen by construction, not by `set_trainable`.

    The activity mask is built by rounding the indicator columns, and `tf.round`
    has zero derivative, so no gradient ever reaches them.  `q_mu` is perturbed
    first because at initialisation (`q_mu = 0`, `q_sqrt = I`) the ELBO is
    stationary in `Z` and *every* column would read zero, making this vacuous.
    """
    model = svgp(create_kernel(kernel_class), Datum.data)
    model.q_mu.assign(Datum.rng.normal(size=model.q_mu.shape))

    Z_variable = model.inducing_variable.Z.unconstrained_variable
    with tf.GradientTape() as tape:
        loss = model.training_loss(Datum.data)
    grad = tape.gradient(loss, Z_variable).numpy()

    np.testing.assert_array_equal(grad[:, INDICATOR_COL], 0.0)
    # The feature columns must be live, otherwise the assertion above is vacuous.
    assert np.any(np.abs(grad[:, FEATURE_COLS]) > 1e-12)


@pytest.mark.parametrize("kernel_class", KERNEL_CLASSES)
def test_inducing_indicator_columns_unchanged_by_training(
    kernel_class: Type[HierarchicalEmbeddingKernel],
) -> None:
    model = svgp(create_kernel(kernel_class), Datum.data)
    Z_before = model.inducing_variable.Z.numpy().copy()

    loss_fn = training_loss_closure(model, Datum.data, compile=True)
    optimiser = tf_keras.optimizers.Adam(learning_rate=0.1)
    for _ in range(ADAM_STEPS):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        optimiser.apply_gradients(
            zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
        )

    Z_after = model.inducing_variable.Z.numpy()

    # Every inducing point is still a legal point of the hierarchical space...
    np.testing.assert_array_equal(Z_after[:, INDICATOR_COL], Z_before[:, INDICATOR_COL])
    # ...while its feature coordinates were free to optimise.
    assert np.any(np.abs(Z_after[:, FEATURE_COLS] - Z_before[:, FEATURE_COLS]) > 1e-6)
