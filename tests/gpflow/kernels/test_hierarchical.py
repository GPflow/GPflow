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

from typing import Mapping, Sequence

import numpy as np
import pytest
import tensorflow as tf

from gpflow.kernels.hierarchical import (
    ActivityCondition,
    ArcHierarchical,
    HierarchicalEmbeddingKernel,
)


class _StubKernel(HierarchicalEmbeddingKernel):
    """Test-only concrete subclass with a deterministic conditional embedding.

    Each conditional column is mapped to ``[v * m, 0]`` — enough to exercise
    the base-class plumbing without bringing in Arc/Wedge maths.
    """

    def _embed_conditional(self, v_c: tf.Tensor, m_c: tf.Tensor) -> tf.Tensor:
        zero = tf.zeros_like(v_c)
        return tf.concat([v_c * m_c, zero], axis=-1)


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
        kernel = _StubKernel(
            feature_dims=[0, 1],
            feature_bounds=_canonical_bounds(2),
        )
        assert kernel._n_feat == 2
        assert kernel._n_uncond == 2
        assert kernel._n_cond == 0
        assert kernel._n_ind == 0

    def test_mixed_conditional_and_unconditional_columns(self) -> None:
        kernel = _StubKernel(
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
    def test_init_rejects_invalid_arguments(
        self, kwargs: dict, match: str
    ) -> None:
        n_feat = len(kwargs.get("feature_dims", []))
        kwargs.setdefault("feature_bounds", _canonical_bounds(max(n_feat, 1)))
        with pytest.raises(ValueError, match=match):
            _StubKernel(**kwargs)

    def test_init_rejects_malformed_feature_bounds_shape(self) -> None:
        with pytest.raises(ValueError, match="feature_bounds"):
            _StubKernel(
                feature_dims=[0, 1],
                feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),  # only 1 row
            )

    def test_init_rejects_inverted_feature_bounds(self) -> None:
        with pytest.raises(ValueError, match="feature_bounds"):
            _StubKernel(
                feature_dims=[0],
                feature_bounds=tf.constant([[1.0, 0.0]], dtype=tf.float64),  # lower > upper
            )


class TestNormalise:
    def test_maps_bounds_to_unit_interval(self) -> None:
        kernel = _StubKernel(
            feature_dims=[0, 1],
            feature_bounds=tf.constant([[0.0, 10.0], [-1.0, 1.0]], dtype=tf.float64),
        )
        X = tf.constant([[5.0, 0.0], [10.0, 1.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        np.testing.assert_allclose(v.numpy(), [[0.5, 0.5], [1.0, 1.0]])

    def test_ignores_non_feature_columns(self) -> None:
        kernel = _StubKernel(
            feature_dims=[0, 2],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 4.0]], dtype=tf.float64),
            indicator_dims=[1],
        )
        X = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        np.testing.assert_allclose(v.numpy(), [[0.5, 0.5]])

    def test_zero_range_bound_does_not_nan(self) -> None:
        kernel = _StubKernel(
            feature_dims=[0],
            feature_bounds=tf.constant([[3.0, 3.0]], dtype=tf.float64),
        )
        X = tf.constant([[3.0], [3.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        assert np.all(np.isfinite(v.numpy()))


def _canonical_disjunction_kernel() -> _StubKernel:
    """4-variable disjunction: x1 unconditional, y1 indicator, x2 active when
    y1=1, x3 active when y1=0. Column layout in X: [x1, y1, x2, x3]."""
    return _StubKernel(
        feature_dims=[0, 2, 3],
        feature_bounds=tf.constant(
            [[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=tf.float64
        ),
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
        np.testing.assert_array_equal(
            mask, [[True, True, False], [True, False, True]]
        )

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
        np.testing.assert_array_equal(
            mask, [[True, True, False], [True, False, True]]
        )

    def test_no_indicators_means_all_active(self) -> None:
        kernel = _StubKernel(
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
        kernel = _StubKernel(
            feature_dims=[0, 1],
            feature_bounds=_canonical_bounds(2),
        )
        X = tf.constant([[0.5, 0.7]], dtype=tf.float64)
        Z = kernel._embed(X)
        assert Z.shape == (1, 2)
        np.testing.assert_allclose(Z.numpy(), [[0.5, 0.7]])

    def test_inactive_column_zeroes_via_stub_embedding(self) -> None:
        kernel = _canonical_disjunction_kernel()
        # y1=0 → x2 inactive → stub embeds it to [0, 0].
        X = tf.constant([[0.5, 0.0, 2.5, 0.5]], dtype=tf.float64)
        Z = kernel._embed(X).numpy()
        # Layout: [x1_normalised, x2_part1, x3_part1, x2_part2, x3_part2]
        # stub conditional embedding: [v*m, 0]
        # x2: m=0 → 0; x3: m=1, v=(0.5-(-1))/2=0.75 → 0.75
        # uncond x1: (0.5-0)/1 = 0.5
        np.testing.assert_allclose(Z, [[0.5, 0.0, 0.75, 0.0, 0.0]])


class TestBaseKernelPlumbing:
    def test_default_base_kernel_is_matern52(self) -> None:
        import gpflow

        kernel = _StubKernel(
            feature_dims=[0], feature_bounds=_canonical_bounds(1)
        )
        assert isinstance(kernel.base_kernel, gpflow.kernels.Matern52)

    def test_base_kernel_lengthscales_frozen_to_one(self) -> None:
        import gpflow

        base = gpflow.kernels.SquaredExponential(lengthscales=3.7)
        kernel = _StubKernel(
            feature_dims=[0], feature_bounds=_canonical_bounds(1), base_kernel=base
        )
        np.testing.assert_allclose(kernel.base_kernel.lengthscales.numpy(), 1.0)
        assert not kernel.base_kernel.lengthscales.trainable

    def test_non_stationary_base_kernel_rejected(self) -> None:
        import gpflow

        with pytest.raises(ValueError, match="Stationary"):
            _StubKernel(
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
        feature_bounds=tf.constant(
            [[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=tf.float64
        ),
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
        K_self = float(kernel.K(X_active, X_active).numpy())
        K_cross = float(kernel.K(X_active, X_inactive).numpy())
        assert K_cross < K_self - 1e-6
