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

from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import pytest
import tensorflow as tf

from gpflow.base import AnyNDArray, Parameter
from gpflow.kernels import Constant, Matern52, SquaredExponential
from gpflow.kernels.hierarchical import (
    ActivityCondition,
    ArcHierarchical,
    HierarchicalEmbeddingKernel,
    HierarchyNode,
    WedgeHierarchical,
)


def _canonical_bounds(n_feat: int) -> tf.Tensor:
    return tf.constant([[0.0, 1.0]] * n_feat, dtype=tf.float64)


def _unconditional_hierarchy(feature_dims: List[int]) -> List[HierarchyNode]:
    """One unconditional node owning every supplied feature column."""
    return [
        HierarchyNode(
            name="all",
            feature_dims=feature_dims,
            feature_bounds=_canonical_bounds(len(feature_dims)),
        ),
    ]


def _canonical_disjunction_hierarchy() -> List[HierarchyNode]:
    """4-variable disjunction: x1 unconditional, y1 indicator (column 1), x2
    active when y1=1, x3 active when y1=0. Column layout in X: [x1, y1, x2, x3]."""
    return [
        HierarchyNode(
            "shared",
            feature_dims=[0],
            feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        ),
        HierarchyNode(
            "branch_A",
            feature_dims=[2],
            feature_bounds=tf.constant([[0.0, 5.0]], dtype=tf.float64),
            activity_condition=ActivityCondition({1: 1}),
        ),
        HierarchyNode(
            "branch_B",
            feature_dims=[3],
            feature_bounds=tf.constant([[-1.0, 1.0]], dtype=tf.float64),
            activity_condition=ActivityCondition({1: 0}),
        ),
    ]


def _canonical_disjunction_kernel() -> ArcHierarchical:
    return ArcHierarchical(
        hierarchy=_canonical_disjunction_hierarchy(),
        active_dims=list(range(4)),
    )


def _all_conditional_hierarchy() -> List[HierarchyNode]:
    """All-conditional 3-column hierarchy: indicator at column 0, x1 at column 1
    active when y0=1, x2 at column 2 active when y0=0. No unconditional features."""
    return [
        HierarchyNode(
            "branch_A",
            feature_dims=[1],
            feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
            activity_condition=ActivityCondition({0: 1}),
        ),
        HierarchyNode(
            "branch_B",
            feature_dims=[2],
            feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
            activity_condition=ActivityCondition({0: 0}),
        ),
    ]


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


class TestHierarchyNode:
    def test_default_activity_condition_is_unconditional(self) -> None:
        node = HierarchyNode("n", feature_dims=[0], feature_bounds=[[0.0, 1.0]])
        assert node.activity_condition == ActivityCondition()

    def test_fields_are_preserved(self) -> None:
        ac = ActivityCondition({0: 1})
        node = HierarchyNode(
            "branch_A",
            feature_dims=[2, 3],
            feature_bounds=[[0.0, 1.0], [-1.0, 1.0]],
            activity_condition=ac,
        )
        assert node.name == "branch_A"
        assert list(node.feature_dims) == [2, 3]
        assert node.activity_condition is ac

    def test_empty_feature_dims_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            HierarchyNode("n", feature_dims=[], feature_bounds=tf.zeros((0, 2), dtype=tf.float64))

    def test_negative_feature_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            HierarchyNode("n", feature_dims=[-1], feature_bounds=[[0.0, 1.0]])

    def test_duplicate_feature_dims_within_node_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            HierarchyNode(
                "n",
                feature_dims=[0, 0],
                feature_bounds=[[0.0, 1.0], [0.0, 1.0]],
            )

    def test_feature_bounds_row_count_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="feature_bounds"):
            HierarchyNode("n", feature_dims=[0, 1], feature_bounds=[[0.0, 1.0]])

    def test_feature_bounds_wrong_rank_rejected(self) -> None:
        with pytest.raises(ValueError, match="feature_bounds"):
            HierarchyNode("n", feature_dims=[0], feature_bounds=[0.0, 1.0])

    def test_inverted_feature_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="lower <= upper"):
            HierarchyNode("n", feature_dims=[0], feature_bounds=[[1.0, 0.0]])

    def test_non_activity_condition_rejected(self) -> None:
        with pytest.raises(ValueError, match="ActivityCondition"):
            HierarchyNode(
                "n",
                feature_dims=[0],
                feature_bounds=[[0.0, 1.0]],
                activity_condition={0: 1},  # type: ignore[arg-type]
            )

    def test_non_string_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            HierarchyNode(
                name=123,  # type: ignore[arg-type]
                feature_dims=[0],
                feature_bounds=[[0.0, 1.0]],
            )

    def test_value_equality(self) -> None:
        a = HierarchyNode("n", feature_dims=[0], feature_bounds=[[0.0, 1.0]])
        b = HierarchyNode("n", feature_dims=[0], feature_bounds=[[0.0, 1.0]])
        c = HierarchyNode("n", feature_dims=[1], feature_bounds=[[0.0, 1.0]])
        assert a == b
        assert a != c


class TestHierarchicalEmbeddingKernelConstruction:
    def test_unconditional_kernel_has_no_conditional_columns(self) -> None:
        kernel = ArcHierarchical(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
        assert kernel._n_feat == 2
        assert kernel._n_uncond == 2
        assert kernel._n_cond == 0
        assert kernel._n_ind == 0

    def test_mixed_conditional_and_unconditional_columns(self) -> None:
        kernel = ArcHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            active_dims=list(range(4)),
        )
        assert kernel._n_feat == 3
        assert kernel._n_uncond == 1
        assert kernel._n_cond == 2
        assert kernel._n_ind == 1
        assert kernel.indicator_dims == (1,)
        assert kernel._uncond_local_idx == [0]
        assert kernel._cond_local_idx == [1, 2]

    def test_features_grouped_by_node_share_activity_condition(self) -> None:
        # A single node owning two features ⇒ both share the same condition.
        kernel = ArcHierarchical(
            hierarchy=[
                HierarchyNode(
                    "branch_A",
                    feature_dims=[0, 2],
                    feature_bounds=_canonical_bounds(2),
                    activity_condition=ActivityCondition({1: 1}),
                ),
            ],
            active_dims=[0, 1, 2],
        )
        assert kernel._n_cond == 2
        assert kernel._cond_local_idx == [0, 1]
        assert kernel.indicator_dims == (1,)

    def test_empty_hierarchy_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one node"):
            ArcHierarchical(hierarchy=[], active_dims=[0])

    def test_duplicate_node_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate node names"):
            ArcHierarchical(
                hierarchy=[
                    HierarchyNode("foo", feature_dims=[0], feature_bounds=[[0.0, 1.0]]),
                    HierarchyNode("foo", feature_dims=[1], feature_bounds=[[0.0, 1.0]]),
                ],
                active_dims=[0, 1],
            )

    def test_duplicate_feature_dims_across_nodes_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ArcHierarchical(
                hierarchy=[
                    HierarchyNode("a", feature_dims=[0], feature_bounds=[[0.0, 1.0]]),
                    HierarchyNode("b", feature_dims=[0], feature_bounds=[[0.0, 1.0]]),
                ],
                active_dims=[0, 1],
            )

    def test_feature_indicator_overlap_rejected(self) -> None:
        # Column 0 is declared both as a feature and as the indicator gating
        # another feature — must be rejected at construction.
        with pytest.raises(ValueError, match="overlap"):
            ArcHierarchical(
                hierarchy=[
                    HierarchyNode("a", feature_dims=[0], feature_bounds=[[0.0, 1.0]]),
                    HierarchyNode(
                        "b",
                        feature_dims=[1],
                        feature_bounds=[[0.0, 1.0]],
                        activity_condition=ActivityCondition({0: 1}),
                    ),
                ],
                active_dims=[0, 1],
            )

    def test_hierarchy_is_stored_for_introspection(self) -> None:
        hierarchy = _canonical_disjunction_hierarchy()
        kernel = ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
        assert kernel._hierarchy == tuple(hierarchy)

    def test_indicator_dims_property_is_derived_and_sorted(self) -> None:
        kernel = ArcHierarchical(
            hierarchy=[
                HierarchyNode("shared", feature_dims=[0], feature_bounds=[[0.0, 1.0]]),
                HierarchyNode(
                    "branch_A",
                    feature_dims=[2],
                    feature_bounds=[[0.0, 1.0]],
                    activity_condition=ActivityCondition({4: 1, 1: 0}),
                ),
                HierarchyNode(
                    "branch_B",
                    feature_dims=[3],
                    feature_bounds=[[0.0, 1.0]],
                    activity_condition=ActivityCondition({4: 0}),
                ),
            ],
            active_dims=list(range(5)),
        )
        # Derived from the union of ActivityCondition keys, sorted ascending.
        assert kernel.indicator_dims == (1, 4)

    def test_required_active_dims_uniform_length_check(self) -> None:
        # The canonical disjunction needs exactly 4 sliced columns; passing 3
        # is rejected regardless of whether active_dims is a list or a slice.
        hierarchy = _canonical_disjunction_hierarchy()
        with pytest.raises(ValueError, match="selects 3"):
            ArcHierarchical(hierarchy=hierarchy, active_dims=[0, 1, 2])
        with pytest.raises(ValueError, match="selects 3"):
            ArcHierarchical(hierarchy=hierarchy, active_dims=slice(0, 3))

    def test_open_ended_active_dims_slice_rejected(self) -> None:
        # A slice whose `stop` is None has no statically determinable width
        # and must be rejected (length check cannot run).
        with pytest.raises(ValueError, match="width"):
            ArcHierarchical(
                hierarchy=_canonical_disjunction_hierarchy(),
                active_dims=slice(None, None, None),
            )

    def test_non_contiguous_sliced_dims_rejected(self) -> None:
        # width matches n_feat + n_ind, but the feature/indicator columns do
        # not form the contiguous range 0..width-1 in the sliced coordinate
        # system (here feature_dims=[0, 3] with width 2) -> rejected.
        with pytest.raises(ValueError, match="sliced coordinate system"):
            ArcHierarchical(
                hierarchy=[
                    HierarchyNode(
                        "a",
                        feature_dims=[0, 3],
                        feature_bounds=_canonical_bounds(2),
                    ),
                ],
                active_dims=[0, 1],
            )

    def test_public_dimension_and_hierarchy_properties(self) -> None:
        hierarchy = _canonical_disjunction_hierarchy()
        kernel = ArcHierarchical(hierarchy=hierarchy, active_dims=list(range(4)))
        # Public read-only views over the construction-time counts/structure.
        assert kernel.n_uncond_dims == 1
        assert kernel.n_cond_dims == 2
        assert tuple(kernel.hierarchy) == tuple(hierarchy)


class TestNormalise:
    def test_maps_bounds_to_unit_interval(self) -> None:
        kernel = ArcHierarchical(
            hierarchy=[
                HierarchyNode(
                    "n",
                    feature_dims=[0, 1],
                    feature_bounds=tf.constant([[0.0, 10.0], [-1.0, 1.0]], dtype=tf.float64),
                ),
            ],
            active_dims=[0, 1],
        )
        X = tf.constant([[5.0, 0.0], [10.0, 1.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        np.testing.assert_allclose(v.numpy(), [[0.5, 0.5], [1.0, 1.0]])

    def test_ignores_indicator_columns(self) -> None:
        # `_normalise` should pick out only the feature columns from its
        # input; indicator columns (here column 1, gated by an
        # `ActivityCondition`) must not appear in the normalised output.
        kernel = ArcHierarchical(
            hierarchy=[
                HierarchyNode(
                    "n",
                    feature_dims=[0, 2],
                    feature_bounds=tf.constant([[0.0, 1.0], [0.0, 4.0]], dtype=tf.float64),
                    activity_condition=ActivityCondition({1: 1}),
                ),
            ],
            active_dims=[0, 1, 2],
        )
        X = tf.constant([[0.5, 1.0, 2.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        np.testing.assert_allclose(v.numpy(), [[0.5, 0.5]])

    def test_zero_range_bound_does_not_nan(self) -> None:
        kernel = ArcHierarchical(
            hierarchy=[
                HierarchyNode(
                    "n",
                    feature_dims=[0],
                    feature_bounds=tf.constant([[3.0, 3.0]], dtype=tf.float64),
                ),
            ],
            active_dims=[0],
        )
        X = tf.constant([[3.0], [3.0]], dtype=tf.float64)
        v = kernel._normalise(X)
        assert np.all(np.isfinite(v.numpy()))


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
        kernel = ArcHierarchical(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
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
        kernel = ArcHierarchical(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
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

        kernel = ArcHierarchical(hierarchy=_unconditional_hierarchy([0]), active_dims=[0])
        assert isinstance(kernel.base_kernel, Matern52)

    def test_base_kernel_lengthscales_frozen_to_one(self) -> None:

        base = SquaredExponential(lengthscales=3.7)
        kernel = ArcHierarchical(
            hierarchy=_unconditional_hierarchy([0]),
            base_kernel=base,
            active_dims=[0],
        )
        np.testing.assert_allclose(kernel.base_kernel.lengthscales.numpy(), 1.0)
        assert not kernel.base_kernel.lengthscales.trainable

    def test_non_stationary_base_kernel_rejected(self) -> None:

        with pytest.raises(ValueError, match="Stationary"):
            ArcHierarchical(
                hierarchy=_unconditional_hierarchy([0]),
                base_kernel=Constant(),
                active_dims=[0],
            )

    def test_supplied_base_kernel_is_not_mutated(self) -> None:
        # Canonical disjunction: n_cond=2, n_uncond=1, D_embed = 2*2 + 1 = 5.
        base = Matern52(lengthscales=2.5)
        original_value = base.lengthscales.numpy()
        original_trainable = base.lengthscales.trainable

        kernel = ArcHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            base_kernel=base,
            active_dims=list(range(4)),
        )

        # The caller's object is untouched.
        np.testing.assert_allclose(base.lengthscales.numpy(), original_value)
        assert base.lengthscales.trainable is original_trainable
        # The kernel holds a different object whose lengthscales are forced.
        assert kernel.base_kernel is not base
        np.testing.assert_allclose(kernel.base_kernel.lengthscales.numpy(), 1.0)
        assert not kernel.base_kernel.lengthscales.trainable

    @pytest.mark.parametrize("bad_lengthscales", [[1.0, 2.0], np.ones(7)])
    def test_base_kernel_wrong_shape_lengthscales_rejected(self, bad_lengthscales: Any) -> None:
        # Canonical disjunction has D_embed=5; shapes (2,) and (7,) are wrong.
        with pytest.raises(ValueError, match="lengthscales"):
            ArcHierarchical(
                hierarchy=_canonical_disjunction_hierarchy(),
                base_kernel=Matern52(lengthscales=bad_lengthscales),
                active_dims=list(range(4)),
            )

    def test_base_kernel_ard_correct_shape_accepted(self) -> None:
        # D_embed = 5 for the canonical disjunction.
        kernel = ArcHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            base_kernel=Matern52(lengthscales=np.linspace(0.5, 2.5, 5)),
            active_dims=list(range(4)),
        )
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
        assert np.all(np.isfinite(K))
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8


class TestUncondLengthscales:
    def test_uncond_lengthscales_exists_and_trainable(self) -> None:
        # Canonical disjunction: n_uncond=1.
        kernel = _canonical_disjunction_kernel()
        assert isinstance(kernel.uncond_lengthscales, Parameter)
        assert tuple(kernel.uncond_lengthscales.shape) == (1,)
        assert kernel.uncond_lengthscales.trainable

        # All-unconditional with two features: n_uncond=2.
        kernel2 = ArcHierarchical(
            hierarchy=_unconditional_hierarchy([0, 1]),
            active_dims=[0, 1],
        )
        assert isinstance(kernel2.uncond_lengthscales, Parameter)
        assert tuple(kernel2.uncond_lengthscales.shape) == (2,)
        assert kernel2.uncond_lengthscales.trainable

    def test_uncond_lengthscales_init_from_vector_lengthscales(self) -> None:
        # d_embed = 5; the first n_uncond=1 entries seed uncond_lengthscales.
        ls = np.linspace(0.5, 2.5, 5)
        kernel = ArcHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            base_kernel=Matern52(lengthscales=ls),
            active_dims=list(range(4)),
        )
        assert kernel.uncond_lengthscales is not None
        np.testing.assert_allclose(kernel.uncond_lengthscales.numpy(), ls[:1])

    def test_uncond_lengthscales_init_from_scalar_defaults_to_ones(self) -> None:
        kernel = ArcHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            base_kernel=Matern52(lengthscales=3.7),
            active_dims=list(range(4)),
        )
        assert kernel.uncond_lengthscales is not None
        np.testing.assert_allclose(kernel.uncond_lengthscales.numpy(), [1.0])

    def test_uncond_lengthscales_absent_when_no_uncond(self) -> None:
        kernel_no_uncond = ArcHierarchical(
            hierarchy=_all_conditional_hierarchy(),
            active_dims=list(range(3)),
        )
        assert kernel_no_uncond.uncond_lengthscales is None

        kernel_with_uncond = _canonical_disjunction_kernel()
        assert (
            len(kernel_with_uncond.trainable_variables)
            == len(kernel_no_uncond.trainable_variables) + 1
        )

    def test_K_off_diagonals_increase_with_uncond_lengthscales(self) -> None:
        # Two points differing only in the unconditional feature (column 0).
        # Increasing uncond_lengthscales shrinks the effective input distance,
        # so the off-diagonal kernel value should move towards the diagonal.
        kernel = _canonical_disjunction_kernel()
        X = tf.constant(
            [
                [0.0, 1.0, 2.5, 0.0],
                [1.0, 1.0, 2.5, 0.0],
            ],
            dtype=tf.float64,
        )
        K_small = kernel.K(X).numpy()
        diag = K_small[0, 0]
        off_small = K_small[0, 1]

        assert kernel.uncond_lengthscales is not None
        kernel.uncond_lengthscales.assign([10.0])
        K_large = kernel.K(X).numpy()
        off_large = K_large[0, 1]

        # Diagonal unchanged (point vs itself).
        np.testing.assert_allclose(K_large[0, 0], diag)
        # Off-diagonal moves closer to the diagonal.
        assert off_large > off_small
        assert off_large < diag

    def test_base_kernel_lengthscales_still_frozen_to_one(self) -> None:
        # The new Parameter must not disturb the base kernel invariant.
        base = SquaredExponential(lengthscales=3.7)
        kernel = ArcHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            base_kernel=base,
            active_dims=list(range(4)),
        )
        np.testing.assert_allclose(kernel.base_kernel.lengthscales.numpy(), 1.0)
        assert not kernel.base_kernel.lengthscales.trainable


def _scatter_columns(
    X_compact: tf.Tensor,
    positions: List[int],
    total_cols: int,
    fill: float = np.nan,
) -> tf.Tensor:
    """Embed a compact ``[N, d]`` array into a wider ``[N, total_cols]`` one.

    Column ``j`` of ``X_compact`` is placed at column ``positions[j]``; every other
    column is set to ``fill``. The default NaN fill turns any accidental read of an
    unselected (redundant) column into a NaN in the output, so a leak is impossible
    to miss.
    """
    X_np = np.asarray(X_compact, dtype=np.float64)
    n_rows, n_cols = X_np.shape
    assert len(positions) == n_cols, "positions must give one target per compact column"
    padded = np.full((n_rows, total_cols), fill, dtype=np.float64)
    for j, p in enumerate(positions):
        padded[:, p] = X_np[:, j]
    return tf.constant(padded, dtype=tf.float64)


def _disjunction_data() -> tf.Tensor:
    """Compact ``[x1, y1, x2, x3]`` data exercising both branches (y1=1 and y1=0)."""
    return tf.constant(
        [
            [0.5, 1.0, 2.5, 0.0],
            [0.3, 0.0, 4.0, 0.5],
            [0.8, 1.0, 1.0, -0.5],
        ],
        dtype=tf.float64,
    )


def _all_conditional_data() -> tf.Tensor:
    """Compact ``[y0, x1, x2]`` data for the all-conditional hierarchy."""
    return tf.constant(
        [
            [1.0, 0.4, 0.7],
            [0.0, 0.2, 0.9],
            [1.0, 0.6, 0.1],
        ],
        dtype=tf.float64,
    )


class TestNonTrivialActiveDims:
    """`active_dims` that do not start at 0 and/or have gaps, with redundant columns.

    The hierarchy's feature/indicator dims live in the *sliced* coordinate system, so
    these tests keep the hierarchy fixed and vary only the physical layout of ``X`` and
    ``active_dims``. Slicing happens in ``Kernel.__call__``, so they go through the full
    ``kernel(X)`` call path (unlike the rest of the suite, which calls ``K`` on
    pre-sliced inputs). A padded/offset layout must reproduce the compact baseline.
    """

    @staticmethod
    def _assert_matches_compact(
        hierarchy_fn: Any,
        X_compact: tf.Tensor,
        positions: List[int],
        total_cols: int,
        active_dims: Any,
    ) -> None:
        n_cols = int(X_compact.shape[1])
        kernel_compact = ArcHierarchical(hierarchy=hierarchy_fn(), active_dims=list(range(n_cols)))
        kernel_padded = ArcHierarchical(hierarchy=hierarchy_fn(), active_dims=active_dims)
        X_padded = _scatter_columns(X_compact, positions, total_cols)

        # Full covariance through the slicing call path.
        K_ref = kernel_compact(X_compact).numpy()
        K_padded = kernel_padded(X_padded).numpy()
        np.testing.assert_allclose(K_padded, K_ref, atol=1e-12)
        # NaN-filled redundant columns never leaked into the computation.
        assert np.all(np.isfinite(K_padded))

        # Diagonal (full_cov=False) through the same call path.
        diag_ref = kernel_compact(X_compact, full_cov=False).numpy()
        diag_padded = kernel_padded(X_padded, full_cov=False).numpy()
        np.testing.assert_allclose(diag_padded, diag_ref, atol=1e-12)
        assert np.all(np.isfinite(diag_padded))

    def test_gap_active_dims_matches_compact(self) -> None:
        # Relevant columns scattered to odd positions in a 9-column array.
        self._assert_matches_compact(
            _canonical_disjunction_hierarchy,
            _disjunction_data(),
            positions=[1, 3, 5, 7],
            total_cols=9,
            active_dims=[1, 3, 5, 7],
        )

    def test_active_dims_not_starting_at_zero(self) -> None:
        # Contiguous block offset by two redundant leading columns.
        self._assert_matches_compact(
            _canonical_disjunction_hierarchy,
            _disjunction_data(),
            positions=[2, 3, 4, 5],
            total_cols=6,
            active_dims=[2, 3, 4, 5],
        )

    def test_scrambled_layout_reconstructed_by_active_dims(self) -> None:
        # Physical layout scrambled; active_dims order must gather back to
        # [x1, y1, x2, x3] so that feature/indicator re-indexing stays correct.
        self._assert_matches_compact(
            _canonical_disjunction_hierarchy,
            _disjunction_data(),
            positions=[3, 0, 5, 1],
            total_cols=6,
            active_dims=[3, 0, 5, 1],
        )

    def test_slice_active_dims_matches_compact(self) -> None:
        # Exercise the `slice` branch of Kernel.slice (distinct from tf.gather).
        self._assert_matches_compact(
            _canonical_disjunction_hierarchy,
            _disjunction_data(),
            positions=[2, 3, 4, 5],
            total_cols=6,
            active_dims=slice(2, 6),
        )

    def test_all_conditional_non_trivial_active_dims(self) -> None:
        # n_uncond == 0: exercises the indicator-gather / activity-mask path under
        # an offset, guarding indicator indexing specifically.
        self._assert_matches_compact(
            _all_conditional_hierarchy,
            _all_conditional_data(),
            positions=[1, 4, 6],
            total_cols=8,
            active_dims=[1, 4, 6],
        )


class TestTFFunctionCompilation:
    """The kernel must be usable inside a ``tf.function`` (graph mode).

    GPflow models wrap loss/prediction in ``tf.function``; the kernel's Python-level
    ``if``/``while`` branch only on static values (``self._n_*`` ints, ``.shape.ndims``),
    so it should trace cleanly. These tests prove it does and that compiled output matches
    eager. An ``input_signature`` with ``None`` rows means a single trace must handle any
    ``N``; a tracing failure raises, so a passing call is itself evidence of compilation.
    """

    def test_K_compiles_and_matches_eager(self) -> None:
        kernel = _canonical_disjunction_kernel()

        @tf.function(input_signature=[tf.TensorSpec([None, 4], dtype=tf.float64)])
        def compiled_K(X: tf.Tensor) -> tf.Tensor:
            return kernel(X)

        for X in (_disjunction_data(), _disjunction_data()[:2]):
            np.testing.assert_allclose(compiled_K(X).numpy(), kernel(X).numpy(), atol=1e-12)

    def test_K_with_X2_compiles_and_matches_eager(self) -> None:
        kernel = _canonical_disjunction_kernel()

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, 4], dtype=tf.float64),
                tf.TensorSpec([None, 4], dtype=tf.float64),
            ]
        )
        def compiled_K(X: tf.Tensor, X2: tf.Tensor) -> tf.Tensor:
            return kernel(X, X2)

        X = _disjunction_data()
        X2 = _disjunction_data()[:2]
        np.testing.assert_allclose(compiled_K(X, X2).numpy(), kernel(X, X2).numpy(), atol=1e-12)

    def test_K_diag_compiles_and_matches_eager(self) -> None:
        kernel = _canonical_disjunction_kernel()

        @tf.function(input_signature=[tf.TensorSpec([None, 4], dtype=tf.float64)])
        def compiled_K_diag(X: tf.Tensor) -> tf.Tensor:
            return kernel(X, full_cov=False)

        for X in (_disjunction_data(), _disjunction_data()[:2]):
            np.testing.assert_allclose(
                compiled_K_diag(X).numpy(),
                kernel(X, full_cov=False).numpy(),
                atol=1e-12,
            )

    def test_batched_input_compiles(self) -> None:
        # Leading batch dim exercises the static-rank rank-padding `while` loop in
        # `_build_activity_mask` under graph mode.
        kernel = _canonical_disjunction_kernel()

        @tf.function(input_signature=[tf.TensorSpec([None, None, 4], dtype=tf.float64)])
        def compiled_K(X: tf.Tensor) -> tf.Tensor:
            return kernel(X)

        X = tf.stack([_disjunction_data(), _disjunction_data() + 0.1], axis=0)
        np.testing.assert_allclose(compiled_K(X).numpy(), kernel(X).numpy(), atol=1e-12)

    def test_all_conditional_compiles(self) -> None:
        # n_uncond == 0: traces the indicator-gather / activity-mask branch.
        kernel = ArcHierarchical(hierarchy=_all_conditional_hierarchy(), active_dims=list(range(3)))

        @tf.function(input_signature=[tf.TensorSpec([None, 3], dtype=tf.float64)])
        def compiled_K(X: tf.Tensor) -> tf.Tensor:
            return kernel(X)

        for X in (_all_conditional_data(), _all_conditional_data()[:2]):
            np.testing.assert_allclose(compiled_K(X).numpy(), kernel(X).numpy(), atol=1e-12)


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


def _arc() -> ArcHierarchical:
    return ArcHierarchical(
        hierarchy=_canonical_disjunction_hierarchy(),
        active_dims=list(range(4)),
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

    def test_angle_sigmoid_bounds_are_per_conditional_vectors(self) -> None:
        # Bounds must be [n_cond] vectors so a multistart scheme that samples
        # `tf.random.uniform(bijector.low.shape, ...)` gets a per-column restart.
        kernel = _arc()
        transform = kernel.angle.transform
        assert transform is not None
        low = transform.low
        high = transform.high
        assert tuple(low.shape) == (2,)
        assert tuple(high.shape) == (2,)
        np.testing.assert_allclose(low.numpy(), [0.1, 0.1])
        np.testing.assert_allclose(high.numpy(), [0.9, 0.9])

    def test_angle_multistart_restart_roundtrip(self) -> None:
        kernel = _arc()
        param = kernel.angle
        transform = param.transform
        assert transform is not None
        low, high = transform.low, transform.high
        sample = tf.random.uniform(low.shape, minval=low, maxval=high, dtype=low.dtype)
        assert tuple(sample.shape) == (2,)
        param.assign(sample)
        assert tuple(param.shape) == (2,)
        assert np.all(param.numpy() >= low.numpy())
        assert np.all(param.numpy() <= high.numpy())
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.5, 0.0, 2.5, 0.5],
                [0.3, 1.0, 4.0, 0.0],
            ],
            dtype=tf.float64,
        )
        K = kernel.K(X).numpy()
        assert np.all(np.isfinite(K))
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8

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
        kernel = ArcHierarchical(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
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


def _wedge() -> WedgeHierarchical:
    return WedgeHierarchical(
        hierarchy=_canonical_disjunction_hierarchy(),
        active_dims=list(range(4)),
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

    def test_rho_sigmoid_bounds_are_per_conditional_vectors(self) -> None:
        # Bounds must be [n_cond] vectors so a multistart scheme that samples
        # `tf.random.uniform(bijector.low.shape, ...)` gets a per-column restart.
        kernel = _wedge()
        transform = kernel.rho.transform
        assert transform is not None
        low = transform.low
        high = transform.high
        assert tuple(low.shape) == (2,)
        assert tuple(high.shape) == (2,)
        np.testing.assert_allclose(low.numpy(), [1e-6, 1e-6])
        np.testing.assert_allclose(high.numpy(), [np.pi, np.pi])

    def test_rho_multistart_restart_roundtrip(self) -> None:
        kernel = _wedge()
        param = kernel.rho
        transform = param.transform
        assert transform is not None
        low, high = transform.low, transform.high
        sample = tf.random.uniform(low.shape, minval=low, maxval=high, dtype=low.dtype)
        assert tuple(sample.shape) == (2,)
        param.assign(sample)
        assert tuple(param.shape) == (2,)
        assert np.all(param.numpy() >= low.numpy())
        assert np.all(param.numpy() <= high.numpy())
        X = tf.constant(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.5, 0.0, 2.5, 0.5],
                [0.3, 1.0, 4.0, 0.0],
            ],
            dtype=tf.float64,
        )
        K = kernel.K(X).numpy()
        assert np.all(np.isfinite(K))
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8

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
        # The reference helpers operate on the flattened per-feature
        # representation that the kernel builds internally.
        return dict(
            feature_dims=[0, 2, 3],
            feature_bounds=np.array([[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=np.float64),
            activity_conditions=[{}, {1: 1}, {1: 0}],
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


# Layout -> (X_padded, active_dims) such that `X_padded[:, active_dims] == X_base`.
# The hierarchy used in TestActiveDims is `_canonical_disjunction_hierarchy()`,
# whose columns are [x1, y1, x2, x3] in the *sliced* coordinate system. Each
# layout below inserts deterministic-but-non-trivial garbage columns around
# (or between) those four columns so that a broken `active_dims` path would
# pick up the garbage.
def _padded_X_and_active_dims(
    layout: str, X_base: AnyNDArray
) -> Tuple[AnyNDArray, Union[List[int], slice]]:
    rng = np.random.default_rng(0)
    N, D = X_base.shape
    if layout == "list_left":
        garbage = rng.standard_normal((N, 2))
        return np.concatenate([garbage, X_base], axis=1), list(range(2, 2 + D))
    if layout == "list_right":
        garbage = rng.standard_normal((N, 3))
        return np.concatenate([X_base, garbage], axis=1), list(range(D))
    if layout == "list_interspersed":
        garbage = rng.standard_normal((N, D + 1))
        cols: List[AnyNDArray] = []
        active_dims: List[int] = []
        position = 0
        for i in range(D):
            cols.append(garbage[:, i : i + 1])
            position += 1
            cols.append(X_base[:, i : i + 1])
            active_dims.append(position)
            position += 1
        cols.append(garbage[:, D : D + 1])
        return np.concatenate(cols, axis=1), active_dims
    if layout == "slice_block":
        garbage = rng.standard_normal((N, 2))
        return np.concatenate([garbage, X_base], axis=1), slice(2, 2 + D)
    raise AssertionError(f"unknown layout {layout!r}")


_ACTIVE_DIMS_LAYOUTS = ["list_left", "list_right", "list_interspersed", "slice_block"]


class TestActiveDims:
    """`active_dims` interpretation contract for hierarchical kernels.

    The canonical disjunction hierarchy is defined in the *sliced* coordinate
    system; padding `X` with garbage columns and setting `active_dims` to skip
    them must produce the same Gram matrix as feeding the unpadded `X`.
    """

    @staticmethod
    def _X_base() -> AnyNDArray:
        return np.array(
            [
                [0.5, 1.0, 2.5, 0.0],
                [0.2, 0.0, 0.0, 0.5],
                [0.7, 1.0, 4.0, -0.4],
            ]
        )

    @staticmethod
    def _ref_args() -> Dict[str, Any]:
        return dict(
            feature_dims=[0, 2, 3],
            feature_bounds=np.array([[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=np.float64),
            activity_conditions=[{}, {1: 1}, {1: 0}],
        )

    @pytest.mark.parametrize("layout", _ACTIVE_DIMS_LAYOUTS)
    def test_arc_matches_reference_under_active_dims(self, layout: str) -> None:
        from tests.gpflow.kernels.reference import ref_arc_hierarchical_kernel

        X_base = self._X_base()
        X_padded, active_dims = _padded_X_and_active_dims(layout, X_base)
        kernel = ArcHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            active_dims=active_dims,
        )
        ref = ref_arc_hierarchical_kernel(
            X_base,
            angle=kernel.angle.numpy(),
            radius=kernel.radius.numpy(),
            **self._ref_args(),
        )
        actual = kernel(tf.constant(X_padded)).numpy()
        np.testing.assert_allclose(actual, ref, rtol=1e-10)

    @pytest.mark.parametrize("layout", _ACTIVE_DIMS_LAYOUTS)
    def test_wedge_matches_reference_under_active_dims(self, layout: str) -> None:
        from tests.gpflow.kernels.reference import ref_wedge_hierarchical_kernel

        X_base = self._X_base()
        X_padded, active_dims = _padded_X_and_active_dims(layout, X_base)
        kernel = WedgeHierarchical(
            hierarchy=_canonical_disjunction_hierarchy(),
            active_dims=active_dims,
        )
        ref = ref_wedge_hierarchical_kernel(
            X_base,
            theta1=kernel.theta1.numpy(),
            theta2=kernel.theta2.numpy(),
            rho=kernel.rho.numpy(),
            **self._ref_args(),
        )
        actual = kernel(tf.constant(X_padded)).numpy()
        np.testing.assert_allclose(actual, ref, rtol=1e-10)

    @pytest.mark.parametrize("kernel_name", ["arc", "wedge"])
    def test_kdiag_matches_diag_of_K_under_active_dims(self, kernel_name: str) -> None:
        X_padded, active_dims = _padded_X_and_active_dims("list_interspersed", self._X_base())
        hierarchy = _canonical_disjunction_hierarchy()
        kernel: HierarchicalEmbeddingKernel
        if kernel_name == "arc":
            kernel = ArcHierarchical(hierarchy=hierarchy, active_dims=active_dims)
        else:
            kernel = WedgeHierarchical(hierarchy=hierarchy, active_dims=active_dims)
        X_tf = tf.constant(X_padded)
        K_full = kernel(X_tf).numpy()
        K_diag = kernel(X_tf, full_cov=False).numpy()
        np.testing.assert_allclose(K_diag, np.diag(K_full), rtol=1e-10)

    @pytest.mark.parametrize("kernel_name", ["arc", "wedge"])
    def test_presliced_roundtrip(self, kernel_name: str) -> None:
        X_padded, active_dims = _padded_X_and_active_dims("list_left", self._X_base())
        hierarchy = _canonical_disjunction_hierarchy()
        kernel: HierarchicalEmbeddingKernel
        if kernel_name == "arc":
            kernel = ArcHierarchical(hierarchy=hierarchy, active_dims=active_dims)
        else:
            kernel = WedgeHierarchical(hierarchy=hierarchy, active_dims=active_dims)
        X_tf = tf.constant(X_padded)
        K_via_call = kernel(X_tf).numpy()
        X_sliced, _ = kernel.slice(X_tf, None)
        K_presliced = kernel(X_sliced, presliced=True).numpy()
        np.testing.assert_allclose(K_via_call, K_presliced, rtol=1e-12)

    def test_active_dims_too_narrow_to_cover_feature_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="active_dims"):
            ArcHierarchical(
                hierarchy=_unconditional_hierarchy([0, 1]),
                active_dims=[0],
            )

    def test_active_dims_too_wide_raises(self) -> None:
        # hierarchy needs 2 feature dims + 0 indicator dims = 2 total;
        # passing 3 active dims must be rejected at construction time.
        with pytest.raises(ValueError, match="active_dims"):
            ArcHierarchical(
                hierarchy=_unconditional_hierarchy([0, 1]),
                active_dims=[0, 1, 2],
            )

    def test_active_dims_exact_count_accepted(self) -> None:
        # 2 feature dims, 0 indicator dims → exactly 2 active_dims required.
        kernel = ArcHierarchical(
            hierarchy=_unconditional_hierarchy([0, 1]),
            active_dims=[3, 5],
        )
        assert kernel is not None


class TestEmptyCases:
    def test_uncond_only_arc_skips_conditional_parameters(self) -> None:
        kernel = ArcHierarchical(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
        X = tf.constant([[0.5, 0.7], [0.3, 0.4], [0.9, 0.1]], dtype=tf.float64)
        K = kernel.K(X).numpy()
        assert K.shape == (3, 3)
        eigs = np.linalg.eigvalsh(K + 1e-10 * np.eye(3))
        assert eigs.min() > -1e-8

    def test_cond_only_wedge(self) -> None:
        kernel = WedgeHierarchical(
            hierarchy=[
                HierarchyNode(
                    "branch_A",
                    feature_dims=[1],
                    feature_bounds=[[0.0, 1.0]],
                    activity_condition=ActivityCondition({0: 1}),
                ),
                HierarchyNode(
                    "branch_B",
                    feature_dims=[2],
                    feature_bounds=[[0.0, 1.0]],
                    activity_condition=ActivityCondition({0: 0}),
                ),
            ],
            active_dims=[0, 1, 2],
        )
        X = tf.constant([[1.0, 0.5, 0.0], [0.0, 0.0, 0.5]], dtype=tf.float64)
        K = kernel.K(X).numpy()
        assert K.shape == (2, 2)
        assert np.all(np.isfinite(K))


class TestComposability:
    def test_arc_composes_with_constant_for_variance(self) -> None:

        kernel = Constant() * _arc()
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
