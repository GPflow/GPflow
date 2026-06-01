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
"""Runtime validation of the three hierarchical-kernel axioms on
arbitrary kernel compositions.

The three axioms are stated in
``doc/sphinx/notebooks/advanced/hierarchical_kernels.pct.py`` as
properties of a *per-dimension distance* ``d_i`` on a conditional
input dimension ``i``:

1. **both inactive:** ``d_i = 0``;
2. **both active:** ``d_i`` is a function of the difference in feature value;
3. **incomparable** (one active, one inactive): ``d_i`` is positive and
   places the two points in distinct regions of the embedded space.

A composed kernel (``Sum``, ``Product``, scaling, ...) no longer exposes
``d_i`` directly, so this module checks the **kernel-level shadows**:

* Axiom 1 shadow: ``K(x, x) == K(x, x')`` when ``x'`` differs from ``x``
  only on the target feature column and the activity condition is
  violated on both sides.
* Axiom 2 shadow: ``K`` is stationary in the target feature column when
  the activity condition is satisfied on both sides — i.e.
  ``K(x_a, y_a) == K(x_b, y_b)`` whenever ``x_a, y_a`` and ``x_b, y_b``
  agree on every other coordinate and their target-column differences
  match.
* Axiom 3 shadow: ``K(x_active, x_inactive) < K(x_active, x_active)``
  when the only change is the activity status of the target feature.

The validator does not walk the kernel tree; it only calls ``kernel.K``,
so it works on any GPflow kernel — including arbitrary
``Sum`` / ``Product`` compositions and future combination types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import tensorflow as tf

from ..base import AnyNDArray
from ..experimental.utils import experimental
from .base import ActiveDims, Kernel
from .hierarchical import ActivityCondition, HierarchyNode


@dataclass(frozen=True)
class AxiomCheck:
    """Outcome of one axiom check on one conditional feature column.

    :param node_name: name of the :class:`HierarchyNode` whose conditional
        behaviour was tested.
    :param feature_dim: column index (sliced coordinate system) of the
        conditional feature dimension that was tested.
    :param axiom: which axiom — ``1``, ``2``, or ``3``.
    :param passed: ``True`` iff the predicate held within tolerance over
        every sampled input.
    :param max_violation: largest violation observed across samples
        (always ``>= 0``). For axioms 1 and 2 this is ``|K_a - K_b|``;
        for axiom 3 this is ``max(0, K_cross - K_self)`` — i.e. by how
        much an incomparable pair was at least as similar as a self pair.
    :param detail: one-line human-readable explanation.
    """

    node_name: str
    feature_dim: int
    axiom: int
    passed: bool
    max_violation: float
    detail: str


@dataclass(frozen=True)
class AxiomReport:
    """Aggregate report from :func:`validate_hierarchical_axioms`.

    Use :attr:`passed` for the boolean summary, :meth:`for_axiom` to slice
    by axiom number, and ``str(report)`` for a human-readable table.
    """

    checks: Tuple[AxiomCheck, ...]

    @property
    def passed(self) -> bool:
        """``True`` iff every axiom check passed (vacuously ``True`` when
        the hierarchy has no conditional nodes)."""
        return all(c.passed for c in self.checks)

    def for_axiom(self, n: int) -> Tuple[AxiomCheck, ...]:
        """Return only the checks for axiom ``n`` (``1``, ``2``, or ``3``)."""
        return tuple(c for c in self.checks if c.axiom == n)

    def __str__(self) -> str:
        if not self.checks:
            return "AxiomReport: no conditional nodes to check (PASS by vacuity)"
        header = f"AxiomReport (overall {'PASS' if self.passed else 'FAIL'})"
        rows = [header]
        for c in self.checks:
            rows.append(
                f"  {'PASS' if c.passed else 'FAIL'} axiom {c.axiom} "
                f"node={c.node_name!r} feature_dim={c.feature_dim} "
                f"max_violation={c.max_violation:.3e} -- {c.detail}"
            )
        return "\n".join(rows)


def _violating_value(required: int) -> int:
    """An integer guaranteed not to equal ``required`` after rounding to
    int. ``ActivityCondition`` requires values be non-negative ints, so
    ``required + 1`` always differs."""
    return required + 1


def _set_active_at(x: AnyNDArray, requirements_full: Dict[int, int]) -> AnyNDArray:
    """Return a copy of ``x`` with every full-input column listed in
    ``requirements_full`` set to its required integer value (satisfies the
    activity condition the mapping was derived from)."""
    out = x.copy()
    for full_pos, required in requirements_full.items():
        out[full_pos] = float(required)
    return out


def _set_inactive_at(x: AnyNDArray, requirements_full: Dict[int, int]) -> AnyNDArray:
    """Return a copy of ``x`` with one full-input column set to a value that
    violates the activity condition the mapping was derived from; the other
    listed columns are still set to their required values, so the flip is
    isolated to a single indicator."""
    out = x.copy()
    items = list(requirements_full.items())
    flip_pos, flip_required = items[0]
    out[flip_pos] = float(_violating_value(flip_required))
    for full_pos, required in items[1:]:
        out[full_pos] = float(required)
    return out


def _resolve_active_dims_mapping(
    active_dims: Optional[ActiveDims],
    max_hierarchy_position: int,
    input_dim: Optional[int],
) -> Tuple[Dict[int, int], int]:
    """Resolve ``active_dims`` into a ``{sliced_position: full_position}``
    mapping table and the implied ``input_dim``.

    ``active_dims`` semantics mirror :class:`gpflow.kernels.Kernel`: ``None``
    means identity, a ``slice`` selects a sub-range of the input, and a
    sequence of ``int`` lists the full-input columns in sliced-position
    order. The returned mapping covers every sliced position the hierarchy
    references (``0..max_hierarchy_position`` inclusive). The returned
    ``input_dim`` is either the caller-supplied value or a default inferred
    from ``active_dims``.
    """
    n_sliced_required = max_hierarchy_position + 1 if max_hierarchy_position >= 0 else 0

    if active_dims is None:
        mapping = {i: i for i in range(n_sliced_required)}
        resolved_input_dim = input_dim if input_dim is not None else max(n_sliced_required, 1)
        return mapping, resolved_input_dim

    if isinstance(active_dims, slice):
        if input_dim is None:
            if active_dims.stop is None:
                raise ValueError(
                    "`active_dims` is an open-ended slice; `input_dim` must be "
                    "provided so the slice can be resolved."
                )
            input_dim = int(active_dims.stop)
        full_positions = list(range(input_dim))[active_dims]
    else:
        full_positions = [int(d) for d in active_dims]
        if input_dim is None:
            input_dim = (max(full_positions) + 1) if full_positions else 1

    if len(full_positions) < n_sliced_required:
        raise ValueError(
            f"`active_dims` resolves to {len(full_positions)} column(s) "
            f"({full_positions!r}), but the hierarchy references sliced "
            f"position {max_hierarchy_position}; need at least "
            f"{n_sliced_required} columns."
        )

    mapping = {i: full_positions[i] for i in range(n_sliced_required)}
    return mapping, input_dim


def _within(a: float, b: float, *, atol: float, rtol: float) -> Tuple[bool, float]:
    """``(ok, |a - b|)``: ``ok`` iff ``|a - b| <= atol + rtol * max(|a|, |b|)``."""
    diff = abs(a - b)
    tol = atol + rtol * max(abs(a), abs(b))
    return diff <= tol, diff


@experimental
def validate_hierarchical_axioms(
    kernel: Kernel,
    hierarchy: Sequence[HierarchyNode],
    *,
    n_samples: int = 8,
    input_dim: Optional[int] = None,
    active_dims: Optional[ActiveDims] = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    seed: Optional[int] = 0,
) -> AxiomReport:
    """Numerically check that ``kernel`` obeys the three conditional-distance
    axioms (kernel-level shadows) on every conditional :class:`HierarchyNode`
    in ``hierarchy``.

    Test inputs are constructed in the sliced coordinate system used by
    :class:`gpflow.kernels.HierarchicalEmbeddingKernel`. For each conditional
    feature column, three predicates on ``K`` are evaluated over
    ``n_samples`` random backgrounds; the worst violation across samples is
    reported.

    :param kernel: any GPflow :class:`Kernel`. The validator only calls
        ``kernel(...)`` (i.e. :meth:`Kernel.__call__`), never ``kernel.K``
        directly. ``__call__`` is what routes per-child ``active_dims``
        slicing through ``Sum`` / ``Product`` / future combination kernels;
        ``Sum.K`` bypasses that slicing and would feed the raw input to
        every child.
    :param hierarchy: the same :class:`HierarchyNode` sequence the user
        passed (or would pass) to a
        :class:`gpflow.kernels.HierarchicalEmbeddingKernel`. Used only to
        construct test inputs; the kernel itself is treated as a black box.
    :param n_samples: number of random backgrounds per (node, feature, axiom)
        triple. Each background drives independent feature samples.
    :param input_dim: width of the test input vectors. Defaults to the
        smallest value compatible with ``active_dims``: when ``active_dims``
        is ``None``, that is ``max(referenced column) + 1`` across the
        hierarchy; when ``active_dims`` is provided, it is
        ``max(active_dims) + 1`` (or the slice's ``stop``).
    :param active_dims: optional mapping from the hierarchy's sliced-coord
        positions to full-input column indices, matching the ``active_dims``
        of the hierarchical kernel under test. Use this when the kernel
        slices an offset sub-range of a wider input (e.g.
        ``active_dims=[2, 3, 4, 5]`` inside a width-6 input). ``None``
        (the default) is the identity mapping.
    :param atol: absolute tolerance used by the equality predicates
        (axioms 1 and 2) and as the minimum margin for axiom 3.
    :param rtol: relative tolerance used by the equality predicates.
    :param seed: RNG seed for reproducibility. ``None`` uses fresh entropy.
    :returns: an :class:`AxiomReport` whose ``passed`` is ``True`` iff every
        check held. The caller decides whether to assert, log, or display.
    """
    rng = np.random.default_rng(seed)

    per_feature: List[Tuple[str, int, Tuple[float, float], ActivityCondition]] = []
    indicator_dims: Set[int] = set()
    for node in hierarchy:
        bounds = np.asarray(node.feature_bounds, dtype=np.float64)
        for i, fd in enumerate(node.feature_dims):
            per_feature.append(
                (
                    node.name,
                    int(fd),
                    (float(bounds[i, 0]), float(bounds[i, 1])),
                    node.activity_condition,
                )
            )
        for k in node.activity_condition.requirements:
            indicator_dims.add(int(k))

    feature_dim_set = {fd for _, fd, _, _ in per_feature}
    all_hier_positions = feature_dim_set | indicator_dims
    max_hier_pos = max(all_hier_positions) if all_hier_positions else -1

    mapping, input_dim = _resolve_active_dims_mapping(active_dims, max_hier_pos, input_dim)
    if input_dim <= 0:
        raise ValueError(f"`input_dim` must be positive; got {input_dim}.")

    # Bounds keyed by *full-input* column.
    bounds_by_full_col: Dict[int, Tuple[float, float]] = {
        mapping[fd]: bnds for _, fd, bnds, _ in per_feature
    }

    def _sample_background() -> AnyNDArray:
        x = np.zeros(input_dim, dtype=np.float64)
        for full_col, (lo, hi) in bounds_by_full_col.items():
            span = hi - lo
            if span > 0:
                eps = 0.05 * span
                x[full_col] = rng.uniform(lo + eps, hi - eps)
            else:
                x[full_col] = lo
        # Indicators default to 0; they'll be overridden by
        # _set_active_at / _set_inactive_at on the condition under test.
        return x

    def _K(a: AnyNDArray, b: AnyNDArray) -> float:
        # Use ``__call__`` rather than ``K``: combination kernels (``Sum``,
        # ``Product``, ...) route through each child's ``__call__`` which
        # applies the child's ``active_dims`` slice, whereas ``Sum.K``
        # bypasses slicing and passes the raw input to each child.
        ta = tf.constant(a.reshape(1, -1), dtype=tf.float64)
        tb = tf.constant(b.reshape(1, -1), dtype=tf.float64)
        return float(kernel(ta, tb).numpy().item())

    checks: List[AxiomCheck] = []

    for node_name, fd, (lo, hi), condition in per_feature:
        if not condition.requirements:
            continue
        full_fd = mapping[fd]
        # Indicator requirements re-keyed to full-input columns.
        requirements_full: Dict[int, int] = {
            mapping[ind_dim]: required for ind_dim, required in condition.requirements.items()
        }
        span = hi - lo
        eps = 0.05 * span if span > 0 else 0.0

        def _sample_in_range() -> float:
            return float(rng.uniform(lo + eps, hi - eps)) if span > 0 else float(lo)

        # ---- Axiom 1: both inactive, K invariant in target feature value.
        max_vio_1 = 0.0
        a1_passed = True
        for _ in range(n_samples):
            base = _set_inactive_at(_sample_background(), requirements_full)
            v_a = _sample_in_range()
            v_b = _sample_in_range()
            x_a = base.copy()
            x_a[full_fd] = v_a
            x_b = base.copy()
            x_b[full_fd] = v_b
            K_aa = _K(x_a, x_a)
            K_ab = _K(x_a, x_b)
            ok, diff = _within(K_aa, K_ab, atol=atol, rtol=rtol)
            max_vio_1 = max(max_vio_1, diff)
            a1_passed = a1_passed and ok
        checks.append(
            AxiomCheck(
                node_name=node_name,
                feature_dim=fd,
                axiom=1,
                passed=a1_passed,
                max_violation=max_vio_1,
                detail=(
                    f"K(x,x) == K(x,x') when both points violate {node_name!r}'s "
                    f"condition and differ only on feature dim {fd}"
                ),
            )
        )

        # ---- Axiom 2: both active, K stationary in the target feature value.
        max_vio_2 = 0.0
        a2_passed = True
        for _ in range(n_samples):
            base = _set_active_at(_sample_background(), requirements_full)
            if span <= 0:
                # Degenerate bounds: nothing meaningful to test; record a pass.
                continue
            half = 0.5 * span
            v_a = float(rng.uniform(lo + eps, hi - eps - half))
            v_y = float(rng.uniform(lo + eps, hi - eps - half))
            lo_shift = (lo + eps) - min(v_a, v_y)
            hi_shift = (hi - eps) - max(v_a, v_y)
            # Defensive: with v_a, v_y drawn from [L, L + 0.4*span] (L = lo + eps),
            # hi_shift >= 0.5*span > 0 >= lo_shift always, so the shift window is
            # never empty here. Kept in case the sampling bounds above change.
            if hi_shift <= lo_shift:  # pragma: no cover
                continue
            shift = float(rng.uniform(lo_shift, hi_shift))
            x_a = base.copy()
            x_a[full_fd] = v_a
            y_a = base.copy()
            y_a[full_fd] = v_y
            x_b = base.copy()
            x_b[full_fd] = v_a + shift
            y_b = base.copy()
            y_b[full_fd] = v_y + shift
            K_a = _K(x_a, y_a)
            K_b = _K(x_b, y_b)
            ok, diff = _within(K_a, K_b, atol=atol, rtol=rtol)
            max_vio_2 = max(max_vio_2, diff)
            a2_passed = a2_passed and ok
        checks.append(
            AxiomCheck(
                node_name=node_name,
                feature_dim=fd,
                axiom=2,
                passed=a2_passed,
                max_violation=max_vio_2,
                detail=(
                    f"K stationary in feature dim {fd} when {node_name!r}'s "
                    f"condition is satisfied on both sides"
                ),
            )
        )

        # ---- Axiom 3: K(active, inactive) < K(active, active).
        max_vio_3 = 0.0
        a3_passed = True
        for _ in range(n_samples):
            bg = _sample_background()
            v = _sample_in_range()
            x_active = _set_active_at(bg, requirements_full)
            x_active[full_fd] = v
            x_inactive = _set_inactive_at(bg, requirements_full)
            x_inactive[full_fd] = v
            K_self = _K(x_active, x_active)
            K_cross = _K(x_active, x_inactive)
            margin = K_self - K_cross
            tol = atol + rtol * abs(K_self)
            if margin <= tol:
                a3_passed = False
            violation = max(0.0, K_cross - K_self + tol)
            max_vio_3 = max(max_vio_3, violation)
        checks.append(
            AxiomCheck(
                node_name=node_name,
                feature_dim=fd,
                axiom=3,
                passed=a3_passed,
                max_violation=max_vio_3,
                detail=(
                    f"K(x_active, x_inactive) < K(x_active, x_active) when "
                    f"{node_name!r}'s condition is flipped at feature dim {fd}"
                ),
            )
        )

    return AxiomReport(tuple(checks))
