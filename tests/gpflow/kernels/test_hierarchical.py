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

from typing import Any, List, Mapping

import numpy as np
import pytest
import tensorflow as tf
from check_shapes import inherit_check_shapes

from gpflow.kernels import Constant, Matern52, SquaredExponential
from gpflow.kernels.hierarchical import (
    ActivityCondition,
    HierarchicalEmbeddingKernel,
    HierarchyNode,
)


# A minimal concrete subclass used only to exercise the abstract base before
# the first real subclass (`ArcHierarchical`) lands in a later PR. Each
# conditional column is stacked with itself, so the output has the required
# ``2 * D_cond`` width and inactive points (m_c = 0) sit at the origin.
class _FakeEmbedKernel(HierarchicalEmbeddingKernel):
    @inherit_check_shapes
    def _embed_conditional(self, v_c: tf.Tensor, m_c: tf.Tensor) -> tf.Tensor:
        masked = v_c * m_c
        return tf.concat([masked, masked], axis=-1)


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


def _canonical_disjunction_kernel() -> _FakeEmbedKernel:
    return _FakeEmbedKernel(
        hierarchy=_canonical_disjunction_hierarchy(),
        active_dims=list(range(4)),
    )


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


class TestHierarchicalEmbeddingKernelConstruction:
    def test_unconditional_kernel_has_no_conditional_columns(self) -> None:
        kernel = _FakeEmbedKernel(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
        assert kernel._n_feat == 2
        assert kernel._n_uncond == 2
        assert kernel._n_cond == 0
        assert kernel._n_ind == 0

    def test_mixed_conditional_and_unconditional_columns(self) -> None:
        kernel = _FakeEmbedKernel(
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
        kernel = _FakeEmbedKernel(
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
            _FakeEmbedKernel(hierarchy=[], active_dims=[0])

    def test_duplicate_node_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate node names"):
            _FakeEmbedKernel(
                hierarchy=[
                    HierarchyNode("foo", feature_dims=[0], feature_bounds=[[0.0, 1.0]]),
                    HierarchyNode("foo", feature_dims=[1], feature_bounds=[[0.0, 1.0]]),
                ],
                active_dims=[0, 1],
            )

    def test_duplicate_feature_dims_across_nodes_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            _FakeEmbedKernel(
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
            _FakeEmbedKernel(
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
        kernel = _FakeEmbedKernel(hierarchy=hierarchy, active_dims=list(range(4)))
        assert kernel._hierarchy == tuple(hierarchy)

    def test_indicator_dims_property_is_derived_and_sorted(self) -> None:
        kernel = _FakeEmbedKernel(
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
            _FakeEmbedKernel(hierarchy=hierarchy, active_dims=[0, 1, 2])
        with pytest.raises(ValueError, match="selects 3"):
            _FakeEmbedKernel(hierarchy=hierarchy, active_dims=slice(0, 3))

    def test_open_ended_active_dims_slice_rejected(self) -> None:
        # A slice whose `stop` is None has no statically determinable width
        # and must be rejected (length check cannot run).
        with pytest.raises(ValueError, match="width"):
            _FakeEmbedKernel(
                hierarchy=_canonical_disjunction_hierarchy(),
                active_dims=slice(None, None, None),
            )


class TestNormalise:
    def test_maps_bounds_to_unit_interval(self) -> None:
        kernel = _FakeEmbedKernel(
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
        kernel = _FakeEmbedKernel(
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
        kernel = _FakeEmbedKernel(
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
        kernel = _FakeEmbedKernel(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
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
        kernel = _FakeEmbedKernel(hierarchy=_unconditional_hierarchy([0, 1]), active_dims=[0, 1])
        X = tf.constant([[0.5, 0.7]], dtype=tf.float64)
        Z = kernel._embed(X)
        assert Z.shape == (1, 2)
        np.testing.assert_allclose(Z.numpy(), [[0.5, 0.7]])


class TestBaseKernelPlumbing:
    def test_default_base_kernel_is_matern52(self) -> None:
        kernel = _FakeEmbedKernel(hierarchy=_unconditional_hierarchy([0]), active_dims=[0])
        assert isinstance(kernel.base_kernel, Matern52)

    def test_base_kernel_lengthscales_frozen_to_one(self) -> None:
        base = SquaredExponential(lengthscales=3.7)
        kernel = _FakeEmbedKernel(
            hierarchy=_unconditional_hierarchy([0]),
            base_kernel=base,
            active_dims=[0],
        )
        np.testing.assert_allclose(kernel.base_kernel.lengthscales.numpy(), 1.0)
        assert not kernel.base_kernel.lengthscales.trainable

    def test_non_stationary_base_kernel_rejected(self) -> None:
        with pytest.raises(ValueError, match="Stationary"):
            _FakeEmbedKernel(
                hierarchy=_unconditional_hierarchy([0]),
                base_kernel=Constant(),
                active_dims=[0],
            )

    def test_supplied_base_kernel_is_not_mutated(self) -> None:
        # Canonical disjunction: n_cond=2, n_uncond=1, D_embed = 2*2 + 1 = 5.
        base = Matern52(lengthscales=2.5)
        original_value = base.lengthscales.numpy()
        original_trainable = base.lengthscales.trainable

        kernel = _FakeEmbedKernel(
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
            _FakeEmbedKernel(
                hierarchy=_canonical_disjunction_hierarchy(),
                base_kernel=Matern52(lengthscales=bad_lengthscales),
                active_dims=list(range(4)),
            )

    def test_base_kernel_ard_correct_shape_accepted(self) -> None:
        # D_embed = 5 for the canonical disjunction.
        kernel = _FakeEmbedKernel(
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
