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
*active and equal in value*. Great for learning functions for use
in generalised disjunctive programming frameworks
"""

import abc
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes, inherit_check_shapes

from gpflow.experimental.utils import experimental

from ..base import Parameter, TensorType
from ..config import default_float
from ..utilities import deepcopy, positive, set_trainable, to_default_float
from .base import ActiveDims, Kernel, NormalizedActiveDims
from .stationaries import Matern52, Stationary


def _check_non_negative_unique(values: Sequence[int], name: str) -> None:
    for v in values:
        if v < 0:
            raise ValueError(f"`{name}` entries must be non-negative; got {v}.")
    if len(set(values)) != len(values):
        raise ValueError(f"`{name}` contains duplicate entries: {list(values)}.")


def _vector_sigmoid(low: float, high: float, n: int) -> tfp.bijectors.Sigmoid:
    """A ``Sigmoid`` bijector whose ``low``/``high`` are length-``n`` vectors, so
    per-element bounds line up with an ``[n]``-shaped Parameter (e.g. for
    bound-aware multistart restart sampling)."""
    ones = tf.ones(n, dtype=default_float())
    return tfp.bijectors.Sigmoid(
        low=to_default_float(low) * ones,
        high=to_default_float(high) * ones,
    )


def _active_dims_width(active_dims: NormalizedActiveDims) -> Optional[int]:
    """Number of columns selected by ``active_dims``, or ``None`` if it cannot
    be determined without knowing the input dimension (an open-ended slice)."""
    if isinstance(active_dims, np.ndarray):
        return int(active_dims.size)
    if isinstance(active_dims, slice):
        if active_dims.stop is None:
            return None
        start = 0 if active_dims.start is None else active_dims.start
        step = 1 if active_dims.step is None else active_dims.step
        return len(range(start, active_dims.stop, step))
    return None  # pragma: no cover -- unreachable: _normalize_active_dims yields slice|ndarray


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


class HierarchicalEmbeddingKernel(Kernel, metaclass=abc.ABCMeta):
    """Abstract base for kernels that embed conditional features into a stationary
    base kernel's input space, gated by indicator-derived activity masks.

    Concrete subclasses implement two methods:

    - :meth:`_conditional_parameters`, which reads the constrained parameters the
      embedding needs and returns them as a dict. :meth:`K` calls it once and shares
      the result between ``X`` and ``X2``, so each parameter transform runs only once.
    - :meth:`_embed_conditional`, which receives the normalised conditional features
      ``v_c``, their float activity mask ``m_c`` (both ``[batch..., N, D_cond]``) and
      the dict from :meth:`_conditional_parameters`, and returns
      ``[batch..., N, 2 * D_cond]``.

    :param hierarchy: sequence of :class:`HierarchyNode`. Each node binds a
        group of feature columns to an :class:`ActivityCondition` that gates
        them. Feature columns must be globally unique across the hierarchy and
        node names must be unique. The set of indicator columns is derived
        from the union of all :class:`ActivityCondition` keys appearing in
        the hierarchy; no separate ``indicator_dims`` argument is required.
    :param base_kernel: stationary base kernel applied in the joint embedded
        space. Defaults to :class:`Matern52` if omitted. The supplied kernel
        is deep-copied via :func:`gpflow.utilities.deepcopy` before use, so
        the caller's object is never mutated. Its ``lengthscales`` must be
        scalar or have shape ``(2 * n_cond + n_uncond,)`` matching the
        embedded dimension; on the copy they are then forced to 1 and set
        non-trainable. The conditional dimensions carry their scale via the
        per-conditional-column parameters of the concrete subclass (e.g.
        ``angle`` / ``radius`` for ``ArcHierarchical`` or ``theta1`` /
        ``theta2`` for ``WedgeHierarchical``). The unconditional dimensions'
        scale is carried by the learnable :attr:`uncond_lengthscales`
        Parameter (shape ``(n_uncond,)``, positive transform). When
        ``base_kernel.lengthscales`` is supplied with shape
        ``(2 * n_cond + n_uncond,)`` the first ``n_uncond`` entries (which
        match the ``[uncond | cond]`` layout produced by ``_embed``) are used
        as the initial values for :attr:`uncond_lengthscales`; otherwise it
        is initialised to ones. When ``n_uncond == 0`` the attribute is
        ``None``.
    :param active_dims: required. ``feature_dims`` and
        :class:`ActivityCondition` keys are interpreted in the sliced
        coordinate system. When :meth:`__call__` receives inputs, the slice
        defined by ``active_dims`` is applied first, and the resulting
        coordinates are then processed according to the hierarchy. Must
        select exactly ``n_feat + n_ind`` columns, where ``n_feat`` is the
        total number of feature columns in the hierarchy and ``n_ind`` is
        the number of derived indicator columns. May be a sequence of int
        column indices, or a :class:`slice` whose ``stop`` is concrete (so
        its width can be validated at construction).
    :param name: optional kernel name.
    """

    @experimental
    def __init__(
        self,
        hierarchy: Sequence[HierarchyNode],
        base_kernel: Optional[Stationary] = None,
        *,
        active_dims: ActiveDims,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(active_dims=active_dims, name=name)

        hierarchy = tuple(hierarchy)
        if not hierarchy:
            raise ValueError("`hierarchy` must contain at least one node.")
        names = [node.name for node in hierarchy]
        duplicate_names = [name for name, count in Counter(names).items() if count > 1]
        if duplicate_names:
            raise ValueError(f"`hierarchy` contains duplicate node names: {duplicate_names}.")

        flat_feature_dims: List[int] = []
        flat_activity_conditions: List[ActivityCondition] = []
        for node in hierarchy:
            for fd in node.feature_dims:
                flat_feature_dims.append(fd)
                flat_activity_conditions.append(node.activity_condition)

        _check_non_negative_unique(flat_feature_dims, "feature_dims")

        indicator_dims: List[int] = sorted(
            {k for ac in flat_activity_conditions for k in ac.requirements}
        )
        overlap = set(flat_feature_dims).intersection(indicator_dims)
        if overlap:
            raise ValueError(
                f"`ActivityCondition` keys overlap with `feature_dims` on columns "
                f"{sorted(overlap)}; a column cannot be both a feature and an indicator."
            )

        self._n_feat = len(flat_feature_dims)
        self._n_ind = len(indicator_dims)

        n_expected = self._n_feat + self._n_ind
        width = _active_dims_width(self._active_dims)
        if width is None:
            raise ValueError(
                "`active_dims` must allow its width to be determined at "
                "construction: pass a sequence of column indices, or a "
                "`slice` whose `stop` is concrete; "
                f"got {self._active_dims!r}."
            )
        if list(range(width)) != sorted(flat_feature_dims + indicator_dims):
            raise ValueError(
                f"`active_dims` selects {width} columns."
                f" The feature and indicator dims reference the "
                f"sliced coordinate system. Checks on this fail: "
                f"{list(range(width))} != "
                f"{sorted(flat_feature_dims + indicator_dims)}"
            )
        self._indicator_dims_tuple: Tuple[int, ...] = tuple(indicator_dims)
        self._hierarchy = hierarchy

        cond_local_idx = [j for j, ac in enumerate(flat_activity_conditions) if ac.requirements]
        uncond_local_idx = [j for j in range(self._n_feat) if j not in set(cond_local_idx)]

        self._n_cond = len(cond_local_idx)
        self._n_uncond = len(uncond_local_idx)

        # Everything `_embed` needs from the hierarchy is worked out once here rather than on
        # every call. Small batches cost a fixed amount per op, so the embedding is kept to as
        # few ops as possible.
        bounds = tf.concat(
            [
                tf.cast(
                    tf.convert_to_tensor(node.feature_bounds, dtype=tf.float64), default_float()
                )
                for node in hierarchy
            ],
            axis=0,
        )  # [n_feat, 2], rows in `flat_feature_dims` order
        lower = bounds[:, 0]
        span = bounds[:, 1] - bounds[:, 0]
        span = tf.where(tf.abs(span) < 1e-12, tf.ones_like(span), span)
        uncond_idx = tf.constant(uncond_local_idx, dtype=tf.int32)
        cond_idx = tf.constant(cond_local_idx, dtype=tf.int32)
        self._uncond_dims = [flat_feature_dims[j] for j in uncond_local_idx]
        self._uncond_lower = tf.gather(lower, uncond_idx)
        self._uncond_range = tf.gather(span, uncond_idx)
        self._cond_dims = [flat_feature_dims[j] for j in cond_local_idx]
        self._cond_lower = tf.gather(lower, cond_idx)
        self._cond_range = tf.gather(span, cond_idx)

        # The activity mask is built from the distinct (indicator column, required value)
        # pairs across all conditions: `_mask_pair_dims` / `_mask_pair_values` hold the pairs,
        # and `_mask_pair_to_cond[p, c]` is 1 when conditional feature `c` requires pair `p`.
        # A feature is active when none of its required pairs mismatch, which takes a fixed
        # number of ops however many conditions the hierarchy has.
        pairs = sorted(
            {
                pair
                for j in cond_local_idx
                for pair in flat_activity_conditions[j].requirements.items()
            }
        )
        pair_index = {pair: p for p, pair in enumerate(pairs)}
        pair_to_cond = [[0.0] * self._n_cond for _ in pairs]
        for c, j in enumerate(cond_local_idx):
            for pair in flat_activity_conditions[j].requirements.items():
                pair_to_cond[pair_index[pair]][c] = 1.0
        self._mask_pair_dims = [dim for dim, _ in pairs]
        self._mask_pair_values = tf.constant(
            [value for _, value in pairs], dtype=default_float(), shape=[len(pairs)]
        )
        self._mask_pair_to_cond = tf.constant(
            pair_to_cond, dtype=default_float(), shape=[len(pairs), self._n_cond]
        )

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

        self.uncond_lengthscales: Optional[Parameter] = None
        if self._n_uncond > 0:
            ls_np = base_kernel.lengthscales.numpy()
            if ls_shape == (d_embed,):
                uncond_ls_init = ls_np[: self._n_uncond]
            else:
                uncond_ls_init = np.ones(self._n_uncond, dtype=ls_np.dtype)
            self.uncond_lengthscales = Parameter(
                uncond_ls_init,
                transform=positive(),
                name="uncond_lengthscales",
            )

        base_kernel.lengthscales.assign(tf.ones_like(base_kernel.lengthscales))
        set_trainable(base_kernel.lengthscales, False)
        self.base_kernel = base_kernel

    @property
    def hierarchy(self) -> Sequence[HierarchyNode]:
        """The hierarchy defining this kernel's structure."""
        return self._hierarchy

    @property
    def indicator_dims(self) -> Tuple[int, ...]:
        """Column indices (sliced coordinate system) of the indicators referenced
        by any :class:`ActivityCondition` in the hierarchy, sorted ascending."""
        return self._indicator_dims_tuple

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
    )  # D_cond is the number of conditional feature columns
    def _build_activity_mask(self, X: TensorType) -> tf.Tensor:
        """Whether each conditional feature column is active, for each point of ``X``.
        Indicators are rounded to the nearest integer before comparison."""
        X = tf.convert_to_tensor(X)
        if self._n_cond == 0:
            shape = tf.concat([tf.shape(X)[:-1], [0]], axis=0)
            return tf.zeros(shape, dtype=tf.bool)
        indicators = tf.round(tf.gather(X, self._mask_pair_dims, axis=-1))  # [..., N, P]
        mismatches = tf.cast(
            tf.not_equal(indicators, tf.cast(self._mask_pair_values, X.dtype)), X.dtype
        )
        # Number of required pairs each conditional feature fails: [..., N, D_cond].
        n_failed = tf.matmul(mismatches, tf.cast(self._mask_pair_to_cond, X.dtype))
        return n_failed < 0.5

    @abc.abstractmethod
    def _conditional_parameters(self) -> Dict[str, tf.Tensor]:
        """The constrained parameters :meth:`_embed_conditional` needs, each read once.

        Reading a :class:`~gpflow.Parameter` runs its transform, so :meth:`K` reads them once
        and shares them between its two inputs."""

    @abc.abstractmethod
    @check_shapes(
        "v_c: [batch..., N, D_cond]", "m_c: [batch..., N, D_cond]", "return: [batch..., N, D_econd]"
    )  # D_econd = 2*D_cond
    def _embed_conditional(
        self, v_c: tf.Tensor, m_c: tf.Tensor, parameters: Mapping[str, tf.Tensor]
    ) -> tf.Tensor:
        """Map ``[batch..., N, D_cond]`` conditional values and float masks to
        ``[batch..., N, 2 * D_cond]`` embedded coordinates, using the ``parameters``
        returned by :meth:`_conditional_parameters`."""

    def _embedding_parameters(self) -> Dict[str, tf.Tensor]:
        """Every constrained parameter :meth:`_embed` needs, each read once."""
        parameters = dict(self._conditional_parameters()) if self._n_cond > 0 else {}
        if self._n_uncond > 0:
            parameters["uncond_lengthscales"] = tf.convert_to_tensor(self.uncond_lengthscales)
        return parameters

    @check_shapes("X: [batch..., N, D]", "return: [batch..., N, D_e]")  # D_e = 2*D_cond + D_uncond
    def _embed(
        self, X: TensorType, parameters: Optional[Mapping[str, tf.Tensor]] = None
    ) -> tf.Tensor:
        """Embed ``X`` as ``[uncond | cond]`` coordinates for the base kernel. Feature values
        are normalised to ``[0, 1]`` by their bounds; conditional ones are gated by
        :meth:`_build_activity_mask`."""
        X = tf.convert_to_tensor(X)
        if parameters is None:
            parameters = self._embedding_parameters()
        parts = []
        if self._n_uncond > 0:
            v_unc = (
                tf.gather(X, self._uncond_dims, axis=-1) - tf.cast(self._uncond_lower, X.dtype)
            ) / tf.cast(self._uncond_range, X.dtype)
            parts.append(v_unc / parameters["uncond_lengthscales"])
        if self._n_cond > 0:
            v_c = (
                tf.gather(X, self._cond_dims, axis=-1) - tf.cast(self._cond_lower, X.dtype)
            ) / tf.cast(self._cond_range, X.dtype)
            m_c = tf.cast(self._build_activity_mask(X), X.dtype)
            parts.append(self._embed_conditional(v_c, m_c, parameters))
        # HierarchyNode/__init__ guarantee at least one feature column, so parts is non-empty.
        return parts[0] if len(parts) == 1 else tf.concat(parts, axis=-1)

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        """
        Evaluate the kernel function on inputs ``X`` and ``X2``.
        Assumes that X and X2 are already sliced according to active_dims, if applicable.
        The embedding and base kernel evaluation are performed in the joint embedded
        space defined by the hierarchy and indicators.
        """
        parameters = self._embedding_parameters()
        Z = self._embed(X, parameters)
        Z2 = self._embed(X2, parameters) if X2 is not None else None
        return self.base_kernel.K(Z, Z2)

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        """
        Evaluate the kernel function on the diagonal of the covariance matrix for input ``X``.
        Assumes that X is already sliced according to active_dims, if applicable.
        """
        # The base kernel is Stationary, so its diagonal is its variance whatever the
        # (embedded) inputs, and X need not be embedded.
        return self.base_kernel.K_diag(X)


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
        base_kernel: Optional[Stationary] = None,
        *,
        active_dims: ActiveDims,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(hierarchy, base_kernel, active_dims=active_dims, name=name)
        if self._n_cond > 0:
            self.angle = Parameter(
                0.5 * tf.ones(self._n_cond, dtype=default_float()),
                transform=_vector_sigmoid(0.1, 0.9, self._n_cond),
                name="angle",
            )
            self.radius = Parameter(
                tf.ones(self._n_cond, dtype=default_float()),
                transform=positive(),
                name="radius",
            )

    def _conditional_parameters(self) -> Dict[str, tf.Tensor]:
        return {
            "frequency": np.pi * tf.convert_to_tensor(self.angle),
            "radius": tf.convert_to_tensor(self.radius),
        }

    @inherit_check_shapes
    def _embed_conditional(
        self, v_c: tf.Tensor, m_c: tf.Tensor, parameters: Mapping[str, tf.Tensor]
    ) -> tf.Tensor:
        theta = parameters["frequency"] * v_c
        radius_masked = parameters["radius"] * m_c
        return tf.concat([radius_masked * tf.sin(theta), radius_masked * tf.cos(theta)], axis=-1)


class WedgeHierarchical(HierarchicalEmbeddingKernel):
    """The Wedge kernel of Horn et al. (2019), section 4.4.

    Each conditional column ``c`` (normalised value ``v_c`` in ``[0, 1]``,
    activity mask ``m_c``) is mapped into the plane via

    .. math::

        \\phi_c(v_c, m_c) = \\big(
            (\\theta_1 (1 - v_c) + \\theta_2 v_c \\cos\\rho)\\, m_c,\\;
            (\\theta_2 v_c \\sin\\rho)\\, m_c
        \\big).

    Inactive points sit at the origin. Active points sit on the straight
    segment from :math:`(\\theta_1, 0)` (the lower bound, ``v_c = 0``) to
    :math:`\\theta_2 (\\cos\\rho, \\sin\\rho)` (the upper bound,
    ``v_c = 1``). So ``theta1`` is the distance from "inactive" to "active at
    the lower bound", ``theta2`` the distance from "inactive" to "active at
    the upper bound", and ``rho`` the angle between the two. Two active
    points with values ``v_c`` and ``v_c'`` are at distance
    :math:`|v_c - v_c'| \\sqrt{\\theta_1^2 + \\theta_2^2 - 2 \\theta_1
    \\theta_2 \\cos\\rho}`. A stationary base kernel then evaluates
    covariance in the joint embedded space.

    Because ``theta1`` is positive, a point whose feature is active at its
    lower bound is distinct from one whose feature is inactive, which lets
    the kernel model a jump between the two.

    When the three points are collinear the kernel reduces to an imputation
    kernel, i.e. the base kernel applied to ``v_c`` with a fixed value
    substituted for "inactive": at the lower bound as ``theta1`` tends to 0,
    at the upper bound as ``theta2`` tends to 0, outside ``[0, 1]`` as ``rho``
    tends to 0 and inside it as ``rho`` tends to :math:`\\pi`. The parameter
    transforms keep ``theta1`` and ``theta2`` positive and ``rho`` strictly
    between 0 and :math:`\\pi`, so these are limiting cases.

    Reference: D. Horn, J. Stork, N.-J. Schüßler and M. Zaefferer,
    "Surrogates for hierarchical search spaces: the wedge-kernel and an
    automated analysis", GECCO 2019.
    """

    def __init__(
        self,
        hierarchy: Sequence[HierarchyNode],
        base_kernel: Optional[Stationary] = None,
        *,
        active_dims: ActiveDims,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(hierarchy, base_kernel, active_dims=active_dims, name=name)
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
                transform=_vector_sigmoid(1e-6, np.pi, self._n_cond),
                name="rho",
            )

    def _conditional_parameters(self) -> Dict[str, tf.Tensor]:
        # Active points are theta1 * (1, 0) + v * (a, b): the lower bound plus v times the
        # vector from the lower bound to the upper bound.
        theta1 = tf.convert_to_tensor(self.theta1)
        theta2 = tf.convert_to_tensor(self.theta2)
        rho = tf.convert_to_tensor(self.rho)
        return {
            "theta1": theta1,
            "a": theta2 * tf.cos(rho) - theta1,
            "b": theta2 * tf.sin(rho),
        }

    @inherit_check_shapes
    def _embed_conditional(
        self, v_c: tf.Tensor, m_c: tf.Tensor, parameters: Mapping[str, tf.Tensor]
    ) -> tf.Tensor:
        vm = v_c * m_c
        return tf.concat(
            [parameters["theta1"] * m_c + parameters["a"] * vm, parameters["b"] * vm], axis=-1
        )
