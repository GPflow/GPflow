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
"""Hierarchical (conditional / disjunctive) kernels.

This module provides covariance functions that respect an activation
structure on the input space: a point whose conditional feature is
*inactive* is not treated as equivalent to one whose feature is
*active and equal in value*.
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import tensorflow as tf

from ..base import TensorType


def _check_non_negative_unique(values: Sequence[int], name: str) -> None:
    for v in values:
        if v < 0:
            raise ValueError(f"`{name}` entries must be non-negative; got {v}.")
    if len(set(values)) != len(values):
        raise ValueError(f"`{name}` contains duplicate entries: {list(values)}.")


@dataclass(frozen=True)
class ActivityCondition:
    """Conjunction of indicator-equality requirements gating one feature column.

    :param requirements: mapping from indicator column index (in the sliced
        coordinate system, i.e. the same space as
        :attr:`HierarchyNode.feature_dims`) to its required integer value.
        An empty mapping denotes an unconditional column. Keys and values must
        be non-negative ``int``.
    """

    requirements: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.requirements.items():
            if not isinstance(key, int) or isinstance(key, bool) or key < 0:
                raise ValueError(f"`requirements` keys must be non-negative ints; got key {key!r}.")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"`requirements` values must be non-negative ints; got value "
                    f"{value!r} for key {key!r}."
                )


@dataclass(frozen=True)
class HierarchyNode:
    """One node of a hierarchical kernel's search space.

    A node groups one or more feature columns that share the same activation
    condition. Every feature owned by the node is gated by
    :attr:`activity_condition`; the empty default condition makes the node's
    features unconditional.

    :param name: human-readable name of the node; used in error messages and
        for debugging. Must be unique within a hierarchy.
    :param feature_dims: column indices of the (real-valued) features owned by
        this node. Must be non-empty and contain no duplicates.
    :param feature_bounds: tensor of shape ``[len(feature_dims), 2]`` giving
        ``(lower, upper)`` per feature, in the same order as
        :attr:`feature_dims`. Used for normalisation to ``[0, 1]``.
    :param activity_condition: :class:`ActivityCondition` shared by every
        feature this node owns. Defaults to an empty (unconditional) condition.
    """

    name: str
    feature_dims: Sequence[int]
    feature_bounds: TensorType
    activity_condition: ActivityCondition = field(default_factory=ActivityCondition)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError(f"`name` must be a str; got {type(self.name).__name__}.")
        feature_dims = list(self.feature_dims)
        if not feature_dims:
            raise ValueError(f"`feature_dims` of node {self.name!r} must be non-empty.")
        _check_non_negative_unique(feature_dims, f"feature_dims of node {self.name!r}")

        if not isinstance(self.activity_condition, ActivityCondition):
            raise ValueError(
                f"`activity_condition` of node {self.name!r} must be an "
                f"ActivityCondition instance; got "
                f"{type(self.activity_condition).__name__}."
            )

        bounds = tf.convert_to_tensor(self.feature_bounds, dtype=tf.float64)
        if bounds.shape != tf.TensorShape([len(feature_dims), 2]):
            raise ValueError(
                f"`feature_bounds` of node {self.name!r} must have shape "
                f"[len(feature_dims), 2] = [{len(feature_dims)}, 2]; got "
                f"{tuple(bounds.shape)}."
            )
        if bool(tf.reduce_any(bounds[:, 1] < bounds[:, 0]).numpy()):
            raise ValueError(
                f"`feature_bounds` rows of node {self.name!r} must satisfy "
                f"lower <= upper; got at least one inverted row."
            )
