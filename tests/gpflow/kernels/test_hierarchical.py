# Copyright 2026 The GPflow Contributors. All Rights Reserved.
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
"""Tests for ``gpflow.kernels.hierarchical``."""

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pytest
import tensorflow as tf

from gpflow.kernels.hierarchical import (
    ActivityCondition,
    ArcHierarchical,
    HierarchicalEmbeddingKernel,
    WedgeHierarchical,
)


def _canonical_bounds(n_feat: int) -> tf.Tensor:
    return tf.constant([[0.0, 1.0]] * n_feat, dtype=tf.float64)


class TestActivityCondition:
    def test_default_construction_is_unconditional(self) -> None:
        condition = ActivityCondition()
        assert dict(condition.requirements) == {}

    def test_explicit_requirements_are_preserved(self) -> None:
        condition = ActivityCondition({0: 1, 2: 0})
        assert dict(condition.requirements) == {0: 1, 2: 0}

    @pytest.mark.parametrize(
        "bad_requirements",
        [
            {"y1": 1},  # non-int key
            {0: "1"},  # non-int value
            {-1: 1},  # negative key
            {0: -1},  # negative value
        ],
    )
    def test_post_init_rejects_invalid_requirements(
        self, bad_requirements: Mapping[object, object]
    ) -> None:
        with pytest.raises(ValueError, match="requirements"):
            ActivityCondition(bad_requirements)  # type: ignore[arg-type]

    def test_value_equality(self) -> None:
        a = ActivityCondition({0: 1})
        b = ActivityCondition({0: 1})
        c = ActivityCondition({0: 0})
        assert a == b
        assert a != c
        assert ActivityCondition() == ActivityCondition()


class TestHierarchicalEmbeddingKernelConstruction:
    def test_unconditional_kernel_has_no_conditional_columns(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 1],
            feature_bounds=_canonical_bounds(2),
        )
        assert kernel._n_feat == 2
        assert kernel._n_uncond == 2
        assert kernel._n_cond == 0
        assert kernel._n_ind == 0

    def test_mixed_conditional_and_unconditional_columns(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 2, 3],
            feature_bounds=_canonical_bounds(3),
            indicator_dims=[1],
            activity_conditions=[
                ActivityCondition(),  # x0: unconditional
                ActivityCondition({0: 1}),  # x2: active when ind 0 == 1
                ActivityCondition({0: 0}),  # x3: active when ind 0 == 0
            ],
        )
        assert kernel._n_feat == 3
        assert kernel._n_uncond == 1
        assert kernel._n_cond == 2
        assert kernel._n_ind == 1
        assert kernel._uncond_local_idx == [0]
        assert kernel._cond_local_idx == [1, 2]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (
                {"feature_dims": [0, 1], "indicator_dims": [1]},
                "overlap",
            ),
            (
                {"feature_dims": [0, 0]},
                "duplicate",
            ),
            (
                {"feature_dims": [0, 1], "indicator_dims": [2, 2]},
                "duplicate",
            ),
            (
                {"feature_dims": [-1]},
                "negative",
            ),
            (
                {"feature_dims": [0], "indicator_dims": [-2]},
                "negative",
            ),
            (
                {
                    "feature_dims": [0],
                    "activity_conditions": [ActivityCondition(), ActivityCondition()],
                },
                "activity_conditions",
            ),
            (
                {
                    "feature_dims": [0],
                    "indicator_dims": [1],
                    "activity_conditions": [ActivityCondition({5: 1})],
                },
                "indicator",
            ),
        ],
    )
    def test_init_rejects_invalid_arguments(self, kwargs: Dict[str, Any], match: str) -> None:
        kwargs = dict(kwargs)
        n_feat = len(kwargs.get("feature_dims", []))
        kwargs.setdefault("feature_bounds", _canonical_bounds(max(n_feat, 1)))
        with pytest.raises(ValueError, match=match):
            ArcHierarchical(**kwargs)

    def test_init_rejects_malformed_feature_bounds_shape(self) -> None:
        with pytest.raises(ValueError, match="feature_bounds"):
            ArcHierarchical(
                feature_dims=[0, 1],
                feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),  # only 1 row
            )

    def test_init_rejects_inverted_feature_bounds(self) -> None:
        with pytest.raises(ValueError, match="feature_bounds"):
            ArcHierarchical(
                feature_dims=[0],
                feature_bounds=tf.constant([[1.0, 0.0]], dtype=tf.float64),  # lower > upper
            )


class TestNormalise:
    def test_maps_bounds_to_unit_interval(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 1],
            feature_bounds=tf.constant([[0.0, 10.0], [-1.0, 1.0]], dtype=tf.float64),
        )
        X = tf.constant([[5.0, 0.0], [10.0, 1.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        np.testing.assert_allclose(v.numpy(), [[0.5, 0.5], [1.0, 1.0]])

    def test_ignores_non_feature_columns(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 2],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 4.0]], dtype=tf.float64),
            indicator_dims=[1],
        )
        X = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        np.testing.assert_allclose(v.numpy(), [[0.5, 0.5]])

    def test_zero_range_bound_does_not_nan(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0],
            feature_bounds=tf.constant([[3.0, 3.0]], dtype=tf.float64),
        )
        X = tf.constant([[3.0], [3.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        assert np.all(np.isfinite(v.numpy()))


def _canonical_disjunction_kernel() -> ArcHierarchical:
    """4-variable disjunction: x1 unconditional, y1 indicator, x2 active when
    y1=1, x3 active when y1=0. Column layout in X: [x1, y1, x2, x3]."""
    return ArcHierarchical(
        feature_dims=[0, 2, 3],
        feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=tf.float64),
        indicator_dims=[1],
        activity_conditions=[
            ActivityCondition(),
            ActivityCondition({0: 1}),
            ActivityCondition({0: 0}),
        ],
    )


class TestActivityMask:
    def test_canonical_disjunction_truth_table(self) -> None:
        kernel = _canonical_disjunction_kernel()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],  # y1 = 1: x1 + x2 active, x3 inactive
                [0.5, 0.0, 2.5, 0.0],  # y1 = 0: x1 + x3 active, x2 inactive
            ],
            dtype=tf.float64,
        )
        mask = kernel._build_activity_mask(X).numpy()
        np.testing.assert_array_equal(mask, [[True, True, False], [True, False, True]])

    def test_jittered_indicators_round_to_nearest(self) -> None:
        kernel = _canonical_disjunction_kernel()
        X = tf.constant(
            [
                [0.5, 0.999999, 2.5, 0.0],  # ε below 1
                [0.5, 1e-7, 2.5, 0.0],  # ε above 0
            ],
            dtype=tf.float64,
        )
        mask = kernel._build_activity_mask(X).numpy()
        np.testing.assert_array_equal(mask, [[True, True, False], [True, False, True]])

    def test_no_indicators_means_all_active(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 1],
            feature_bounds=_canonical_bounds(2),
        )
        X = tf.constant([[0.5, 0.5], [0.3, 0.7]], dtype=tf.float64)
        mask = kernel._build_activity_mask(X).numpy()
        np.testing.assert_array_equal(mask, np.ones((2, 2), dtype=bool))

    def test_mask_supports_leading_batch_dims(self) -> None:
        # Regression guard: trieste prototype used tf.shape(X)[0], which breaks
        # for [batch..., N, D] inputs.
        kernel = _canonical_disjunction_kernel()
        X = tf.constant(
            [
                [[0.5, 1.0, 2.5, 0.0], [0.5, 0.0, 2.5, 0.0]],
                [[0.5, 0.0, 2.5, 0.0], [0.5, 1.0, 2.5, 0.0]],
            ],
            dtype=tf.float64,
        )
        mask = kernel._build_activity_mask(X).numpy()
        assert mask.shape == (2, 2, 3)
        expected = np.array(
            [
                [[True, True, False], [True, False, True]],
                [[True, False, True], [True, True, False]],
            ]
        )
        np.testing.assert_array_equal(mask, expected)


class TestEmbed:
    def test_shape_mixed_uncond_and_cond(self) -> None:
        kernel = _canonical_disjunction_kernel()
        # 1 uncond column + 2 cond columns => 1 + 2*2 = 5 embedded coords.
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 1.0, 0.5],
            ],
            dtype=tf.float64,
        )
        Z = kernel._embed(X)
        assert Z.shape == (2, 5)

    def test_uncond_only(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 1],
            feature_bounds=_canonical_bounds(2),
        )
        X = tf.constant([[0.5, 0.7]], dtype=tf.float64)
        Z = kernel._embed(X)
        assert Z.shape == (1, 2)
        np.testing.assert_allclose(Z.numpy(), [[0.5, 0.7]])

    def test_inactive_columns_embed_to_origin(self) -> None:
        kernel = _canonical_disjunction_kernel()
        # y1 = 0 ⇒ x2 inactive; y1 = 1 ⇒ x3 inactive.
        X = tf.constant(
            [
                [0.5, 0.0, 2.5, 0.5],  # x2 inactive
                [0.5, 1.0, 2.5, 0.5],  # x3 inactive
            ],
            dtype=tf.float64,
        )
        Z = kernel._embed(X).numpy()
        # Layout: [x1_normalised, sin_x2, sin_x3, cos_x2, cos_x3].
        # Inactive column ⇒ both sin and cos coordinates are 0.
        assert Z.shape == (2, 5)
        np.testing.assert_allclose(Z[0, 1], 0.0)  # sin_x2 row 0 (x2 inactive)
        np.testing.assert_allclose(Z[0, 3], 0.0)  # cos_x2 row 0
        np.testing.assert_allclose(Z[1, 2], 0.0)  # sin_x3 row 1 (x3 inactive)
        np.testing.assert_allclose(Z[1, 4], 0.0)  # cos_x3 row 1
        # Unconditional column passes through normalised.
        np.testing.assert_allclose(Z[:, 0], [0.5, 0.5])


class TestBaseKernelPlumbing:
    def test_default_base_kernel_is_matern52(self) -> None:
        import gpflow

        kernel = ArcHierarchical(feature_dims=[0], feature_bounds=_canonical_bounds(1))
        assert isinstance(kernel.base_kernel, gpflow.kernels.Matern52)

    def test_base_kernel_lengthscales_frozen_to_one(self) -> None:
        import gpflow

        base = gpflow.kernels.SquaredExponential(lengthscales=3.7)
        kernel = ArcHierarchical(
            feature_dims=[0], feature_bounds=_canonical_bounds(1), base_kernel=base
        )
        np.testing.assert_allclose(kernel.base_kernel.lengthscales.numpy(), 1.0)
        assert not kernel.base_kernel.lengthscales.trainable

    def test_non_stationary_base_kernel_rejected(self) -> None:
        import gpflow

        with pytest.raises(ValueError, match="Stationary"):
            ArcHierarchical(
                feature_dims=[0],
                feature_bounds=_canonical_bounds(1),
                base_kernel=gpflow.kernels.Constant(),
            )


class TestKDispatch:
    def test_K_shape_and_psd(self) -> None:
        kernel = _canonical_disjunction_kernel()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.5, 0.0, 2.5, 0.5],
                [0.3, 1.0, 4.0, 0.0],
            ],
            dtype=tf.float64,
        )
        K = kernel.K(X).numpy()
        assert K.shape == (3, 3)
        # Symmetric (X2 omitted ⇒ K(X, X))
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        # PSD: eigenvalues non-negative (allow tiny numerical noise)
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8

    def test_K_with_X2(self) -> None:
        kernel = _canonical_disjunction_kernel()
        X = tf.constant([[0.5, 1.0, 2.5, 0.0]], dtype=tf.float64)
        X2 = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
            ],
            dtype=tf.float64,
        )
        K = kernel.K(X, X2).numpy()
        assert K.shape == (1, 2)

    def test_K_diag_matches_diag_of_K(self) -> None:
        kernel = _canonical_disjunction_kernel()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
                [0.7, 1.0, 4.0, -0.4],
            ],
            dtype=tf.float64,
        )
        np.testing.assert_allclose(
            kernel.K_diag(X).numpy(),
            np.diag(kernel.K(X).numpy()),
            rtol=1e-12,
        )


def _arc(feature_dims: Sequence[int] = (0, 2, 3)) -> ArcHierarchical:
    return ArcHierarchical(
        feature_dims=list(feature_dims),
        feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=tf.float64),
        indicator_dims=[1],
        activity_conditions=[
            ActivityCondition(),
            ActivityCondition({0: 1}),
            ActivityCondition({0: 0}),
        ],
    )


class TestArcHierarchical:
    def test_parameters_are_per_conditional_column(self) -> None:
        kernel = _arc()
        assert tuple(kernel.angle.shape) == (2,)
        assert tuple(kernel.radius.shape) == (2,)

    def test_angle_initialised_in_centre_of_bound(self) -> None:
        kernel = _arc()
        np.testing.assert_allclose(kernel.angle.numpy(), [0.5, 0.5])

    def test_radius_initialised_to_one(self) -> None:
        kernel = _arc()
        np.testing.assert_allclose(kernel.radius.numpy(), [1.0, 1.0])

    def test_K_shape_and_psd(self) -> None:
        kernel = _arc()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.5, 0.0, 2.5, 0.5],
                [0.3, 1.0, 4.0, 0.0],
            ],
            dtype=tf.float64,
        )
        K = kernel.K(X).numpy()
        assert K.shape == (3, 3)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8

    def test_no_conditional_columns_means_no_angle_parameter(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 1],
            feature_bounds=_canonical_bounds(2),
        )
        assert not hasattr(kernel, "angle")
        assert not hasattr(kernel, "radius")

    def test_axiom_both_inactive_is_invariant_in_conditional_value(self) -> None:
        kernel = _arc()
        # y1 = 1 ⇒ x3 is inactive on both rows. Varying x3 must not change K.
        X_a = tf.constant([[0.5, 1.0, 2.5, 0.0]], dtype=tf.float64)
        X_b = tf.constant([[0.5, 1.0, 2.5, 0.7]], dtype=tf.float64)
        K_aa = kernel.K(X_a, X_a).numpy()
        K_ab = kernel.K(X_a, X_b).numpy()
        np.testing.assert_allclose(K_aa, K_ab, rtol=1e-12)

    def test_axiom_one_active_one_inactive_distinct(self) -> None:
        kernel = _arc()
        # Same x1, same x2 value, but y1 flips ⇒ x2 active on one row, x3 on the other.
        X_active = tf.constant([[0.5, 1.0, 2.5, 0.0]], dtype=tf.float64)
        X_inactive = tf.constant([[0.5, 0.0, 2.5, 0.0]], dtype=tf.float64)
        K_self = kernel.K(X_active, X_active).numpy().item()
        K_cross = kernel.K(X_active, X_inactive).numpy().item()
        assert K_cross < K_self - 1e-6


def _wedge(feature_dims: Sequence[int] = (0, 2, 3)) -> WedgeHierarchical:
    return WedgeHierarchical(
        feature_dims=list(feature_dims),
        feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=tf.float64),
        indicator_dims=[1],
        activity_conditions=[
            ActivityCondition(),
            ActivityCondition({0: 1}),
            ActivityCondition({0: 0}),
        ],
    )


class TestWedgeHierarchical:
    def test_parameters_are_per_conditional_column(self) -> None:
        kernel = _wedge()
        assert tuple(kernel.theta1.shape) == (2,)
        assert tuple(kernel.theta2.shape) == (2,)
        assert tuple(kernel.rho.shape) == (2,)

    def test_thetas_initialised_to_one(self) -> None:
        kernel = _wedge()
        np.testing.assert_allclose(kernel.theta1.numpy(), [1.0, 1.0])
        np.testing.assert_allclose(kernel.theta2.numpy(), [1.0, 1.0])

    def test_rho_initialised_to_half_pi(self) -> None:
        kernel = _wedge()
        np.testing.assert_allclose(kernel.rho.numpy(), [np.pi / 2, np.pi / 2])

    def test_K_shape_and_psd(self) -> None:
        kernel = _wedge()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.5, 0.0, 2.5, 0.5],
                [0.3, 1.0, 4.0, 0.0],
            ],
            dtype=tf.float64,
        )
        K = kernel.K(X).numpy()
        assert K.shape == (3, 3)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8

    def test_axiom_both_inactive_is_invariant_in_conditional_value(self) -> None:
        kernel = _wedge()
        X_a = tf.constant([[0.5, 1.0, 2.5, 0.0]], dtype=tf.float64)
        X_b = tf.constant([[0.5, 1.0, 2.5, 0.7]], dtype=tf.float64)
        np.testing.assert_allclose(
            kernel.K(X_a, X_a).numpy(),
            kernel.K(X_a, X_b).numpy(),
            rtol=1e-12,
        )

    def test_axiom_one_active_one_inactive_distinct(self) -> None:
        kernel = _wedge()
        X_active = tf.constant([[0.5, 1.0, 2.5, 0.0]], dtype=tf.float64)
        X_inactive = tf.constant([[0.5, 0.0, 2.5, 0.0]], dtype=tf.float64)
        K_self = kernel.K(X_active, X_active).numpy().item()
        K_cross = kernel.K(X_active, X_inactive).numpy().item()
        assert K_cross < K_self - 1e-6

    def test_axiom_both_active_equal_value_equals_self(self) -> None:
        kernel = _wedge()
        X_a = tf.constant([[0.5, 1.0, 2.5, 0.0]], dtype=tf.float64)
        X_b = tf.constant([[0.5, 1.0, 2.5, 0.4]], dtype=tf.float64)
        # x1 same, y1 same, x2 same; x3 differs but is inactive on both.
        np.testing.assert_allclose(
            kernel.K(X_a, X_a).numpy(),
            kernel.K(X_a, X_b).numpy(),
            rtol=1e-12,
        )


class TestClosedFormReference:
    @staticmethod
    def _ref_args() -> Dict[str, Any]:
        return dict(
            feature_dims=[0, 2, 3],
            feature_bounds=np.array([[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=np.float64),
            indicator_dims=[1],
            activity_conditions=[{}, {0: 1}, {0: 0}],
        )

    def test_arc_matches_numpy_reference(self) -> None:
        from tests.gpflow.kernels.reference import ref_arc_hierarchical_kernel

        kernel = _arc()
        X = np.array(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
                [0.7, 1.0, 4.0, -0.4],
            ]
        )
        ref = ref_arc_hierarchical_kernel(
            X,
            angle=kernel.angle.numpy(),
            radius=kernel.radius.numpy(),
            **self._ref_args(),
        )
        actual = kernel.K(tf.constant(X)).numpy()
        np.testing.assert_allclose(actual, ref, rtol=1e-10)

    def test_wedge_matches_numpy_reference(self) -> None:
        from tests.gpflow.kernels.reference import ref_wedge_hierarchical_kernel

        kernel = _wedge()
        X = np.array(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
                [0.7, 1.0, 4.0, -0.4],
            ]
        )
        ref = ref_wedge_hierarchical_kernel(
            X,
            theta1=kernel.theta1.numpy(),
            theta2=kernel.theta2.numpy(),
            rho=kernel.rho.numpy(),
            **self._ref_args(),
        )
        actual = kernel.K(tf.constant(X)).numpy()
        np.testing.assert_allclose(actual, ref, rtol=1e-10)


class TestEmptyCases:
    def test_uncond_only_arc_skips_conditional_parameters(self) -> None:
        kernel = ArcHierarchical(
            feature_dims=[0, 1],
            feature_bounds=_canonical_bounds(2),
        )
        X = tf.constant([[0.5, 0.7], [0.3, 0.4], [0.9, 0.1]], dtype=tf.float64)
        K = kernel.K(X).numpy()
        assert K.shape == (3, 3)
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8

    def test_cond_only_wedge(self) -> None:
        kernel = WedgeHierarchical(
            feature_dims=[1, 2],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype=tf.float64),
            indicator_dims=[0],
            activity_conditions=[
                ActivityCondition({0: 1}),
                ActivityCondition({0: 0}),
            ],
        )
        X = tf.constant([[1.0, 0.5, 0.0], [0.0, 0.0, 0.5]], dtype=tf.float64)
        K = kernel.K(X).numpy()
        assert K.shape == (2, 2)
        assert np.all(np.isfinite(K))


class TestComposability:
    def test_arc_composes_with_constant_for_variance(self) -> None:
        import gpflow

        kernel = gpflow.kernels.Constant() * _arc()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
            ],
            dtype=tf.float64,
        )
        K = kernel.K(X).numpy()
        assert K.shape == (2, 2)
        assert np.all(np.isfinite(K))


class TestDifferentiability:
    def test_arc_gradients_are_finite_and_nonzero(self) -> None:
        kernel = _arc()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
                [0.7, 1.0, 4.0, -0.4],
                [0.3, 0.0, 0.0, -0.2],
            ],
            dtype=tf.float64,
        )
        with tf.GradientTape() as tape:
            K = tf.reduce_sum(kernel.K(X))
        grads = tape.gradient(
            K,
            [kernel.angle.unconstrained_variable, kernel.radius.unconstrained_variable],
        )
        for g in grads:
            assert g is not None
            g_np = g.numpy()
            assert np.all(np.isfinite(g_np))
            assert np.any(np.abs(g_np) > 1e-12)

    def test_wedge_gradients_are_finite_and_nonzero(self) -> None:
        kernel = _wedge()
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
                [0.7, 1.0, 4.0, -0.4],
                [0.3, 0.0, 0.0, -0.2],
            ],
            dtype=tf.float64,
        )
        with tf.GradientTape() as tape:
            K = tf.reduce_sum(kernel.K(X))
        grads = tape.gradient(
            K,
            [
                kernel.theta1.unconstrained_variable,
                kernel.theta2.unconstrained_variable,
                kernel.rho.unconstrained_variable,
            ],
        )
        for g in grads:
            assert g is not None
            g_np = g.numpy()
            assert np.all(np.isfinite(g_np))
            assert np.any(np.abs(g_np) > 1e-12)
