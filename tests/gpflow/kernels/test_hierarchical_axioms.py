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
"""Tests for ``gpflow.kernels.hierarchical_axioms``."""

import warnings
from typing import List

import pytest
import tensorflow as tf

from gpflow.kernels import (
    ActivityCondition,
    ArcHierarchical,
    Constant,
    HierarchyNode,
    Matern52,
    Polynomial,
    WedgeHierarchical,
    validate_hierarchical_axioms,
)
from gpflow.kernels.hierarchical_axioms import AxiomCheck, AxiomReport


def _canonical_disjunction_hierarchy() -> List[HierarchyNode]:
    """4-variable disjunction in sliced coords: ``[x1, y1, x2, x3]`` — ``x1``
    unconditional, ``y1`` indicator (col 1), ``x2`` active when ``y1=1``,
    ``x3`` active when ``y1=0``. Matches the canonical hierarchy used by
    ``tests/gpflow/kernels/test_hierarchical.py``."""
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


def _arc() -> ArcHierarchical:
    return ArcHierarchical(
        hierarchy=_canonical_disjunction_hierarchy(),
        active_dims=list(range(4)),
    )


def _wedge() -> WedgeHierarchical:
    return WedgeHierarchical(
        hierarchy=_canonical_disjunction_hierarchy(),
        active_dims=list(range(4)),
    )


@pytest.fixture(autouse=True)
def _silence_experimental_warning() -> None:
    """``validate_hierarchical_axioms`` is marked ``@experimental`` and warns
    on first call; that's expected and we don't want to fail tests on it."""
    warnings.filterwarnings(
        "ignore",
        message=r".*validate_hierarchical_axioms.*experimental.*",
    )


# ------------- Passing compositions ---------------------------------------


class TestPassingCompositions:
    def test_arc_alone_passes_all_axioms(self) -> None:
        report = validate_hierarchical_axioms(
            _arc(), _canonical_disjunction_hierarchy(), seed=0
        )
        assert report.passed, str(report)
        # Two conditional nodes × three axioms = 6 checks.
        assert len(report.checks) == 6

    def test_wedge_alone_passes_all_axioms(self) -> None:
        report = validate_hierarchical_axioms(
            _wedge(), _canonical_disjunction_hierarchy(), seed=0
        )
        assert report.passed, str(report)

    def test_constant_times_arc_passes(self) -> None:
        kernel = Constant() * _arc()
        report = validate_hierarchical_axioms(
            kernel, _canonical_disjunction_hierarchy(), seed=0
        )
        assert report.passed, str(report)

    def test_arc_plus_wedge_same_hierarchy_passes(self) -> None:
        kernel = _arc() + _wedge()
        report = validate_hierarchical_axioms(
            kernel, _canonical_disjunction_hierarchy(), seed=0
        )
        assert report.passed, str(report)

    def test_arc_plus_matern_on_unconditional_dim_only_passes(self) -> None:
        # Matern only touches dim 0 (x1, unconditional) — it cannot react to
        # the activity mask or to the conditional feature dims, so the
        # composition inherits the axiom-respecting behaviour of arc.
        kernel = _arc() + Matern52(active_dims=[0])
        report = validate_hierarchical_axioms(
            kernel, _canonical_disjunction_hierarchy(), seed=0
        )
        assert report.passed, str(report)


# ------------- Failing compositions ---------------------------------------


class TestFailingCompositions:
    def test_arc_plus_matern_on_conditional_dim_fails_axiom_1(self) -> None:
        # Matern52 with active_dims that include conditional feature columns
        # responds to changes in those columns regardless of the activity
        # mask. Axiom 1 must flag this.
        kernel = _arc() + Matern52(active_dims=list(range(4)))
        report = validate_hierarchical_axioms(
            kernel, _canonical_disjunction_hierarchy(), seed=0
        )
        assert not report.passed
        a1_checks = report.for_axiom(1)
        assert any(not c.passed for c in a1_checks)
        for c in a1_checks:
            if not c.passed:
                assert c.max_violation > 1e-6

    def test_plain_matern_over_full_space_fails_axiom_1(self) -> None:
        # No hierarchy awareness at all — every axiom-1 check must fail
        # because Matern responds to the conditional feature value when
        # the activity condition is violated.
        kernel = Matern52()
        report = validate_hierarchical_axioms(
            kernel, _canonical_disjunction_hierarchy(), seed=0
        )
        assert not report.passed
        assert all(not c.passed for c in report.for_axiom(1))

    def test_polynomial_plus_arc_fails_axiom_2(self) -> None:
        # Polynomial is non-stationary, so K depends on absolute coordinate
        # values rather than just their differences. Axiom 2 must flag this.
        kernel = _arc() + Polynomial(active_dims=list(range(4)))
        report = validate_hierarchical_axioms(
            kernel, _canonical_disjunction_hierarchy(), seed=0
        )
        assert not report.passed
        a2_checks = report.for_axiom(2)
        assert any(not c.passed for c in a2_checks)

    def test_constant_alone_fails_axiom_3(self) -> None:
        # Constant kernel returns the same variance for every pair, so
        # K(active, inactive) == K(active, active): axiom 3 has zero
        # margin and must fail.
        kernel = Constant()
        report = validate_hierarchical_axioms(
            kernel, _canonical_disjunction_hierarchy(), seed=0
        )
        assert not report.passed
        a3_checks = report.for_axiom(3)
        assert all(not c.passed for c in a3_checks)


# ------------- Report structure & determinism -----------------------------


class TestReportStructure:
    def test_report_breaks_down_per_node_and_per_axiom(self) -> None:
        report = validate_hierarchical_axioms(
            _arc(), _canonical_disjunction_hierarchy(), seed=0
        )
        node_names = {c.node_name for c in report.checks}
        axioms = {c.axiom for c in report.checks}
        assert node_names == {"branch_A", "branch_B"}
        assert axioms == {1, 2, 3}
        # Each conditional node has its single feature dim tested for each axiom.
        per_node = {n: 0 for n in node_names}
        for c in report.checks:
            per_node[c.node_name] += 1
        assert per_node == {"branch_A": 3, "branch_B": 3}

    def test_unconditional_only_hierarchy_passes_vacuously(self) -> None:
        # A hierarchy with no conditional nodes has no axioms to check;
        # the report should be empty and `passed` True by vacuity.
        hierarchy = [
            HierarchyNode(
                "all",
                feature_dims=[0, 1],
                feature_bounds=tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype=tf.float64),
            ),
        ]
        kernel = ArcHierarchical(hierarchy=hierarchy, active_dims=[0, 1])
        report = validate_hierarchical_axioms(kernel, hierarchy, seed=0)
        assert report.checks == ()
        assert report.passed
        assert "no conditional nodes" in str(report)

    def test_str_contains_pass_fail_summary(self) -> None:
        report = validate_hierarchical_axioms(
            _arc(), _canonical_disjunction_hierarchy(), seed=0
        )
        s = str(report)
        assert "PASS" in s
        assert "axiom 1" in s
        assert "axiom 2" in s
        assert "axiom 3" in s

    def test_axiom_check_has_expected_fields(self) -> None:
        check = AxiomCheck(
            node_name="x",
            feature_dim=2,
            axiom=1,
            passed=True,
            max_violation=0.0,
            detail="ok",
        )
        # Frozen dataclass: attribute access works, mutation does not.
        assert check.feature_dim == 2
        with pytest.raises(Exception):
            check.passed = False  # type: ignore[misc]

    def test_axiom_report_is_immutable_tuple_of_checks(self) -> None:
        report = AxiomReport(checks=(
            AxiomCheck("x", 0, 1, True, 0.0, "ok"),
        ))
        assert isinstance(report.checks, tuple)
        assert report.passed


class TestDeterminism:
    def test_same_seed_gives_identical_report(self) -> None:
        h = _canonical_disjunction_hierarchy()
        r1 = validate_hierarchical_axioms(_arc(), h, seed=42)
        r2 = validate_hierarchical_axioms(_arc(), h, seed=42)
        assert r1.checks == r2.checks

    def test_different_seeds_can_give_different_reports(self) -> None:
        # Seed only controls the test-input RNG; the kernel itself is
        # deterministic. With a passing kernel both reports still pass,
        # but the per-check `max_violation` values should differ across
        # seeds because the sampled inputs differ.
        h = _canonical_disjunction_hierarchy()
        r1 = validate_hierarchical_axioms(_arc(), h, seed=1)
        r2 = validate_hierarchical_axioms(_arc(), h, seed=2)
        assert r1.passed and r2.passed
        v1 = tuple(c.max_violation for c in r1.checks)
        v2 = tuple(c.max_violation for c in r2.checks)
        assert v1 != v2


class TestInputDim:
    def test_input_dim_default_uses_max_referenced_column(self) -> None:
        report = validate_hierarchical_axioms(
            _arc(), _canonical_disjunction_hierarchy(), seed=0
        )
        assert report.passed

    def test_explicit_input_dim_overrides_default(self) -> None:
        # Pass a wider input_dim; the kernel still has active_dims=[0..3] so
        # extra columns are ignored. The validator should still pass.
        report = validate_hierarchical_axioms(
            _arc(),
            _canonical_disjunction_hierarchy(),
            seed=0,
            input_dim=6,
        )
        assert report.passed

    def test_negative_input_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            validate_hierarchical_axioms(
                _arc(),
                _canonical_disjunction_hierarchy(),
                seed=0,
                input_dim=-1,
            )
