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

from typing import Mapping

import pytest
import tensorflow as tf

from gpflow.kernels.hierarchical import ActivityCondition, HierarchyNode


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
