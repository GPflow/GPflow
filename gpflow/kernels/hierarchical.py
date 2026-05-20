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
from typing import List, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes, inherit_check_shapes

from gpflow.experimental.utils import experimental

from ..base import Parameter, TensorType
from ..config import default_float
from ..utilities import deepcopy, positive, set_trainable, to_default_float
from .base import ActiveDims, Kernel
from .stationaries import Matern52, Stationary

# Value used to denote inactive or ignored feature entries in
# hierarchical/disjunctive kernel activation logic.
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
        if bounds.shape.rank != 2 or bounds.shape[0] != len(feature_dims) or bounds.shape[1] != 2:
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


class HierarchicalEmbeddingKernel(Kernel, metaclass=abc.ABCMeta):
    """Abstract base for kernels that embed conditional features into a stationary
    base kernel's input space, gated by indicator-derived activity masks.

    Concrete subclasses implement :meth:`_embed_conditional`, which receives the
    normalised conditional features ``v_c`` and their float activity mask ``m_c``
    (both ``[batch..., N, D_cond]``) and returns ``[batch..., N, 2 * D_cond]``.

    :param hierarchy: sequence of :class:`HierarchyNode`. Each node binds a
        group of feature columns to an :class:`ActivityCondition` that gates
        them. Feature columns must be globally unique across the hierarchy and
        node names must be unique.
    :param indicator_dims: column indices of indicator (integer-valued) values
        in ``X``. Indicator indices used in each node's activity condition are
        interpreted as positions within this sequence.
    :param base_kernel: stationary base kernel applied in the joint embedded
        space. Defaults to :class:`Matern52` if omitted. The supplied kernel
        is deep-copied via :func:`gpflow.utilities.deepcopy` before use, so
        the caller's object is never mutated. Its ``lengthscales`` must be
        scalar or have shape ``(2 * n_cond + n_uncond,)`` matching the
        embedded dimension; the copy's lengthscales are then forced to 1
        and set non-trainable, because the per-conditional-column
        parameters of the concrete subclass (e.g. ``angle`` / ``radius``
        for ``ArcHierarchical`` or ``theta1`` / ``theta2`` for
        ``WedgeHierarchical``) already carry the scale of each embedded
        dimension.
    :param active_dims: inherited from :class:`Kernel`. If supplied, both
        feature dims and ``indicator_dims`` are interpreted in the sliced
        coordinate system. When the __call__ method receives inputs, the
        slice defined by ``active_dims`` is applied first, and the resulting
        coordinates are then processed according to the hierarchy and indicators.
    :param name: optional kernel name.
    """

    @experimental
    def __init__(
        self,
        hierarchy: Sequence[HierarchyNode],
        indicator_dims: Sequence[int] = (),
        base_kernel: Optional[Stationary] = None,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(active_dims=active_dims, name=name)

        hierarchy = tuple(hierarchy)
        if not hierarchy:
            raise ValueError("`hierarchy` must contain at least one node.")
        names = [node.name for node in hierarchy]
        if len(set(names)) != len(names):
            raise ValueError(f"`hierarchy` contains duplicate node names: {names}.")

        indicator_dims = list(indicator_dims)
        _check_non_negative_unique(indicator_dims, "indicator_dims")

        for node in hierarchy:
            for k in node.activity_condition.requirements:
                if k >= len(indicator_dims):
                    raise ValueError(
                        f"`hierarchy` node {node.name!r} references indicator "
                        f"index {k}, but only {len(indicator_dims)} indicator "
                        f"dims were declared."
                    )

        flat_feature_dims: List[int] = []
        flat_bounds_rows: List[tf.Tensor] = []
        flat_activity_conditions: List[ActivityCondition] = []
        for node in hierarchy:
            bounds = tf.convert_to_tensor(node.feature_bounds, dtype=tf.float64)
            for i, fd in enumerate(node.feature_dims):
                flat_feature_dims.append(fd)
                flat_bounds_rows.append(bounds[i])
                flat_activity_conditions.append(node.activity_condition)

        _check_non_negative_unique(flat_feature_dims, "feature_dims")
        overlap = set(flat_feature_dims).intersection(indicator_dims)
        if overlap:
            raise ValueError(
                f"`feature_dims` and `indicator_dims` overlap on columns {sorted(overlap)}."
            )

        bounds_tensor = tf.stack(flat_bounds_rows, axis=0)

        self._n_feat = len(flat_feature_dims)
        self._n_ind = len(indicator_dims)

        self._feature_dims = tf.constant(flat_feature_dims, dtype=tf.int32)
        self._indicator_dims = tf.constant(indicator_dims, dtype=tf.int32)
        self._bounds = bounds_tensor
        self._hierarchy = hierarchy

        cond_local_idx = [j for j, ac in enumerate(flat_activity_conditions) if ac.requirements]
        uncond_local_idx = [j for j in range(self._n_feat) if j not in set(cond_local_idx)]

        self._cond_local_idx = cond_local_idx
        self._uncond_local_idx = uncond_local_idx
        self._n_cond = len(cond_local_idx)
        self._n_uncond = len(uncond_local_idx)

        required = [[_IGNORE] * self._n_ind for _ in range(self._n_feat)]
        for j, ac in enumerate(flat_activity_conditions):
            for k, v in ac.requirements.items():
                required[j][k] = v
        self._required = tf.constant(required, dtype=tf.int32)
        self._required_is_ignore = tf.equal(self._required, _IGNORE)

        if base_kernel is None:
            base_kernel = Matern52()
        else:
            if not isinstance(base_kernel, Stationary):
                raise ValueError(
                    f"`base_kernel` must be a gpflow.kernels.Stationary instance; "
                    f"got {type(base_kernel).__name__}."
                )
            base_kernel = deepcopy(base_kernel)

        d_embed = 2 * self._n_cond + self._n_uncond
        ls_shape = tuple(base_kernel.lengthscales.shape)
        if ls_shape not in ((), (1,), (d_embed,)):
            raise ValueError(
                f"`base_kernel.lengthscales` must be scalar or have shape "
                f"({d_embed},) to match the embedded dimension "
                f"`2 * n_cond + n_uncond`; got shape {ls_shape}."
            )

        base_kernel.lengthscales.assign(tf.ones_like(base_kernel.lengthscales))
        set_trainable(base_kernel.lengthscales, False)
        self.base_kernel = base_kernel

    @property
    def hierarchy(self) -> Sequence[HierarchyNode]:
        """The hierarchy defining this kernel's structure."""
        return self._hierarchy

    @property
    def n_cond_dims(self) -> int:
        """Get number of conditional feature dimensions."""
        return self._n_cond

    @property
    def n_uncond_dims(self) -> int:
        """Get number of unconditional feature dimensions."""
        return self._n_uncond

    @check_shapes(
        "X: [batch..., N, D]", "return: [batch..., N, D_cond]"
    )  # D_cond is inferred from context
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

    @check_shapes(
        "X: [batch..., N, D]", "return: [batch..., N, D_f]"
    )  # D_f is feature_dims (i.e. D_cond + D_uncond)
    def _normalise(self, X: TensorType) -> tf.Tensor:
        X_cast = tf.cast(X, self._bounds.dtype)
        v = tf.gather(X_cast, self._feature_dims, axis=-1)
        lo, hi = self._bounds[:, 0], self._bounds[:, 1]
        rng = hi - lo
        safe_rng = tf.where(tf.abs(rng) < 1e-12, tf.ones_like(rng), rng)
        return (v - lo) / safe_rng

    @abc.abstractmethod
    @check_shapes(
        "v_c: [batch..., N, D_cond]", "m_c: [batch..., N, D_cond]", "return: [batch..., N, D_econd]"
    )  # D_econd = 2*D_cond
    def _embed_conditional(self, v_c: tf.Tensor, m_c: tf.Tensor) -> tf.Tensor:
        """Map ``[batch..., N, D_cond]`` conditional values and float masks to
        ``[batch..., N, 2 * D_cond]`` embedded coordinates."""

    @check_shapes("X: [batch..., N, D]", "return: [batch..., N, D_e]")  # D_e = 2*D_cond + D_uncond
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
        if (
            not parts
        ):  # pragma: no subspace activated - unreachable: enforced by HierarchyNode/__init__
            shape = tf.concat([tf.shape(X)[:-1], [0]], axis=0)
            return tf.zeros(shape, dtype=v.dtype)
        return tf.concat(parts, axis=-1)

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        """
        Evaluate the kernel function on inputs ``X`` and ``X2``.
        Assumes that X and X2 are already sliced according to active_dims, if applicable.
        The embedding and base kernel evaluation are performed in the joint embedded
        space defined by the hierarchy and indicators.
        """
        Z = self._embed(X)
        Z2 = self._embed(X2) if X2 is not None else None
        return self.base_kernel.K(Z, Z2)

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        """
        Evaluate the kernel function on the diagonal of the covariance matrix for input ``X``.
        Assumes that X is already sliced according to active_dims, if applicable.
        """
        return self.base_kernel.K_diag(self._embed(X))


class ArcHierarchical(HierarchicalEmbeddingKernel):
    """The Arc kernel of Swersky et al. (2014).

    Each conditional column ``c`` (normalised value ``v_c`` in ``[0, 1]``,
    activity mask ``m_c``) is mapped into the plane via

    .. math::

        \\phi_c(v_c, m_c) = \\big(
            r_c \\sin(\\pi a_c v_c)\\, m_c,\\;
            r_c \\cos(\\pi a_c v_c)\\, m_c
        \\big),

    so that inactive points sit at the origin and active points sit on a
    circle whose phase depends on ``v_c``. A stationary base kernel then
    evaluates covariance in the joint embedded space.
    """

    def __init__(
        self,
        hierarchy: Sequence[HierarchyNode],
        indicator_dims: Sequence[int] = (),
        base_kernel: Optional[Stationary] = None,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(hierarchy, indicator_dims, base_kernel, active_dims=active_dims, name=name)
        if self._n_cond > 0:
            self.angle = Parameter(
                0.5 * tf.ones(self._n_cond, dtype=default_float()),
                transform=tfp.bijectors.Sigmoid(to_default_float(0.1), to_default_float(0.9)),
                name="angle",
            )
            self.radius = Parameter(
                tf.ones(self._n_cond, dtype=default_float()),
                transform=positive(),
                name="radius",
            )

    @inherit_check_shapes
    def _embed_conditional(self, v_c: tf.Tensor, m_c: tf.Tensor) -> tf.Tensor:
        theta = np.pi * self.angle * v_c
        sin_part = self.radius * tf.sin(theta) * m_c
        cos_part = self.radius * tf.cos(theta) * m_c
        return tf.concat([sin_part, cos_part], axis=-1)


class WedgeHierarchical(HierarchicalEmbeddingKernel):
    """The Wedge kernel of Horn et al. (2019).

    Each conditional column ``c`` (normalised value ``v_c`` in ``[0, 1]``,
    activity mask ``m_c``) is mapped into the plane via

    .. math::

        \\phi_c(v_c, m_c) = \\big(
            (\\theta_1 v_c + \\theta_2 v_c \\cos\\rho)\\, m_c,\\;
            (\\theta_2 v_c \\sin\\rho)\\, m_c
        \\big).

    The "incomparable" distance now scales with the active value ``v_c``
    rather than being constant in it. ``rho`` is bounded away from zero
    because at ``rho = 0`` the embedding degenerates to a line.
    """

    def __init__(
        self,
        hierarchy: Sequence[HierarchyNode],
        indicator_dims: Sequence[int] = (),
        base_kernel: Optional[Stationary] = None,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(hierarchy, indicator_dims, base_kernel, active_dims=active_dims, name=name)
        if self._n_cond > 0:
            self.theta1 = Parameter(
                tf.ones(self._n_cond, dtype=default_float()),
                transform=positive(),
                name="theta1",
            )
            self.theta2 = Parameter(
                tf.ones(self._n_cond, dtype=default_float()),
                transform=positive(),
                name="theta2",
            )
            self.rho = Parameter(
                0.5 * np.pi * tf.ones(self._n_cond, dtype=default_float()),
                transform=tfp.bijectors.Sigmoid(to_default_float(1e-6), to_default_float(np.pi)),
                name="rho",
            )

    @inherit_check_shapes
    def _embed_conditional(self, v_c: tf.Tensor, m_c: tf.Tensor) -> tf.Tensor:
        comp1 = (self.theta1 * v_c + self.theta2 * v_c * tf.cos(self.rho)) * m_c
        comp2 = (self.theta2 * v_c * tf.sin(self.rho)) * m_c
        return tf.concat([comp1, comp2], axis=-1)
