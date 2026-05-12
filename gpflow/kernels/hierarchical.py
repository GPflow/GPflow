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

import abc
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import tensorflow as tf

from ..base import TensorType
from ..utilities import set_trainable
from .base import ActiveDims, Kernel
from .stationaries import Matern52, Stationary

_IGNORE = -1


def _check_non_negative_unique(values: Sequence[int], name: str) -> None:
    for v in values:
        if v < 0:
            raise ValueError(f"`{name}` entries must be non-negative; got {v}.")
    if len(set(values)) != len(values):
        raise ValueError(f"`{name}` contains duplicate entries: {list(values)}.")


@dataclass(frozen=True)
class ActivityCondition:
    """Conjunction of indicator-equality requirements gating one feature column.

    :param requirements: mapping from *local indicator index* (the position of
        the indicator within ``indicator_dims``) to its required integer value.
        An empty mapping denotes an unconditional column. Keys and values must
        be non-negative ``int``.
    """

    requirements: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.requirements.items():
            if not isinstance(key, int) or isinstance(key, bool) or key < 0:
                raise ValueError(
                    f"`requirements` keys must be non-negative ints; got key {key!r}."
                )
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"`requirements` values must be non-negative ints; got value "
                    f"{value!r} for key {key!r}."
                )


class HierarchicalEmbeddingKernel(Kernel, metaclass=abc.ABCMeta):
    """Abstract base for kernels that embed conditional features into a stationary
    base kernel's input space, gated by indicator-derived activity masks.

    Concrete subclasses implement :meth:`_embed_conditional`, which receives the
    normalised conditional features ``v_c`` and their float activity mask ``m_c``
    (both ``[batch..., N, D_cond]``) and returns ``[batch..., N, 2 * D_cond]``.

    :param feature_dims: column indices of real-valued (non-indicator) features
        in the flat input ``X``.
    :param feature_bounds: tensor of shape ``[len(feature_dims), 2]`` giving
        ``(lower, upper)`` per feature. Used for normalisation to ``[0, 1]``.
    :param indicator_dims: column indices of indicator (integer-valued) values
        in ``X``.
    :param activity_conditions: one :class:`ActivityCondition` per entry of
        ``feature_dims`` (same order). An empty :class:`ActivityCondition`
        denotes an unconditional column. If omitted, all feature columns are
        treated as unconditional.
    :param active_dims: inherited from :class:`Kernel`. If supplied, both
        ``feature_dims`` and ``indicator_dims`` are interpreted in the sliced
        coordinate system.
    :param name: optional kernel name.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        feature_bounds: TensorType,
        indicator_dims: Sequence[int] = (),
        activity_conditions: Sequence[ActivityCondition] = (),
        base_kernel: Optional[Stationary] = None,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(active_dims=active_dims, name=name)

        feature_dims = list(feature_dims)
        indicator_dims = list(indicator_dims)

        _check_non_negative_unique(feature_dims, "feature_dims")
        _check_non_negative_unique(indicator_dims, "indicator_dims")
        overlap = set(feature_dims).intersection(indicator_dims)
        if overlap:
            raise ValueError(
                f"`feature_dims` and `indicator_dims` overlap on columns "
                f"{sorted(overlap)}."
            )

        bounds_tensor = tf.convert_to_tensor(feature_bounds, dtype=tf.float64)
        if bounds_tensor.shape.rank != 2 or bounds_tensor.shape[0] != len(feature_dims) or bounds_tensor.shape[1] != 2:
            raise ValueError(
                f"`feature_bounds` must have shape [len(feature_dims), 2] = "
                f"[{len(feature_dims)}, 2]; got {tuple(bounds_tensor.shape)}."
            )
        if bool(tf.reduce_any(bounds_tensor[:, 1] < bounds_tensor[:, 0]).numpy()):
            raise ValueError(
                "`feature_bounds` rows must satisfy lower <= upper; "
                "got at least one inverted row."
            )

        if not activity_conditions:
            activity_conditions = [ActivityCondition() for _ in feature_dims]
        else:
            activity_conditions = list(activity_conditions)
            if len(activity_conditions) != len(feature_dims):
                raise ValueError(
                    f"`activity_conditions` must have one entry per "
                    f"`feature_dims` column ({len(feature_dims)}); got "
                    f"{len(activity_conditions)}."
                )

        for j, ac in enumerate(activity_conditions):
            for k in ac.requirements:
                if k >= len(indicator_dims):
                    raise ValueError(
                        f"`activity_conditions[{j}]` references indicator "
                        f"index {k}, but only {len(indicator_dims)} indicator "
                        f"dims were declared."
                    )

        self._n_feat = len(feature_dims)
        self._n_ind = len(indicator_dims)

        self._feature_dims = tf.constant(feature_dims, dtype=tf.int32)
        self._indicator_dims = tf.constant(indicator_dims, dtype=tf.int32)
        self._bounds = bounds_tensor

        cond_local_idx = [
            j for j, ac in enumerate(activity_conditions) if ac.requirements
        ]
        uncond_local_idx = [
            j for j in range(self._n_feat) if j not in set(cond_local_idx)
        ]

        self._cond_local_idx = cond_local_idx
        self._uncond_local_idx = uncond_local_idx
        self._n_cond = len(cond_local_idx)
        self._n_uncond = len(uncond_local_idx)

        required = [[_IGNORE] * self._n_ind for _ in range(self._n_feat)]
        for j, ac in enumerate(activity_conditions):
            for k, v in ac.requirements.items():
                required[j][k] = v
        self._required = tf.constant(required, dtype=tf.int32)
        self._required_is_ignore = tf.equal(self._required, _IGNORE)

        if base_kernel is None:
            base_kernel = Matern52()
        if not isinstance(base_kernel, Stationary):
            raise ValueError(
                f"`base_kernel` must be a gpflow.kernels.Stationary instance; "
                f"got {type(base_kernel).__name__}."
            )
        base_kernel.lengthscales.assign(tf.ones_like(base_kernel.lengthscales))
        set_trainable(base_kernel.lengthscales, False)
        self.base_kernel = base_kernel

    def _build_activity_mask(self, X: TensorType) -> tf.Tensor:
        if self._n_ind == 0:
            shape = tf.concat([tf.shape(X)[:-1], [self._n_feat]], axis=0)
            return tf.ones(shape, dtype=tf.bool)
        ind = tf.gather(X, self._indicator_dims, axis=-1)
        ind = tf.cast(tf.round(tf.cast(ind, tf.float64)), tf.int32)  # [..., N, D_i]
        # broadcast comparison: ind[..., :, None, :] vs _required[None, :, :]
        ind_expanded = tf.expand_dims(ind, axis=-2)  # [..., N, 1, D_i]
        required_expanded = self._required[None, :, :]  # [1, D_f, D_i]
        # left-pad required for extra leading batch dims
        while required_expanded.shape.ndims < ind_expanded.shape.ndims:
            required_expanded = tf.expand_dims(required_expanded, axis=0)
        ignore_expanded = self._required_is_ignore[None, :, :]
        while ignore_expanded.shape.ndims < ind_expanded.shape.ndims:
            ignore_expanded = tf.expand_dims(ignore_expanded, axis=0)
        match = tf.logical_or(
            ignore_expanded,
            tf.equal(ind_expanded, required_expanded),
        )
        return tf.reduce_all(match, axis=-1)

    def _normalise(self, X: TensorType) -> tf.Tensor:
        X_cast = tf.cast(X, self._bounds.dtype)
        v = tf.gather(X_cast, self._feature_dims, axis=-1)
        lo, hi = self._bounds[:, 0], self._bounds[:, 1]
        rng = hi - lo
        safe_rng = tf.where(tf.abs(rng) < 1e-12, tf.ones_like(rng), rng)
        return (v - lo) / safe_rng

    @abc.abstractmethod
    def _embed_conditional(self, v_c: tf.Tensor, m_c: tf.Tensor) -> tf.Tensor:
        """Map ``[batch..., N, D_cond]`` conditional values and float masks to
        ``[batch..., N, 2 * D_cond]`` embedded coordinates."""

    def _embed(self, X: TensorType) -> tf.Tensor:
        v = self._normalise(X)
        m_float = tf.cast(self._build_activity_mask(X), v.dtype)
        parts = []
        if self._n_uncond > 0:
            parts.append(tf.gather(v, self._uncond_local_idx, axis=-1))
        if self._n_cond > 0:
            v_c = tf.gather(v, self._cond_local_idx, axis=-1)
            m_c = tf.gather(m_float, self._cond_local_idx, axis=-1)
            parts.append(self._embed_conditional(v_c, m_c))
        if not parts:
            shape = tf.concat([tf.shape(X)[:-1], [0]], axis=0)
            return tf.zeros(shape, dtype=v.dtype)
        return tf.concat(parts, axis=-1)

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        Z = self._embed(X)
        Z2 = self._embed(X2) if X2 is not None else None
        return self.base_kernel.K(Z, Z2)

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self.base_kernel.K_diag(self._embed(X))
