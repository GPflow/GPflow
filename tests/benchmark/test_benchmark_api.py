# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
from typing import Collection
from unittest.mock import Mock

import pandas as pd
import pytest

import benchmark.datasets as ds
import benchmark.models as md
import benchmark.plotters as pl
from benchmark.benchmark_api import BenchmarkSet, BenchmarkSuite, BenchmarkTask
from benchmark.grouping import GroupingKey as GK
from benchmark.grouping import GroupingSpec


@pytest.mark.parametrize(
    "benchmark_set,expected_tasks",
    [
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                    ds.tiny_sine,
                ],
                models=[
                    md.gpr,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False],
                do_optimise=[False],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            [
                BenchmarkTask(
                    dataset_name="tiny_linear",
                    model_name="gpr",
                    do_compile=False,
                    do_optimise=False,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
                BenchmarkTask(
                    dataset_name="tiny_sine",
                    model_name="gpr",
                    do_compile=False,
                    do_optimise=False,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
            ],
        ),
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                ],
                models=[
                    md.gpr,
                    md.svgp,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False],
                do_optimise=[False],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            [
                BenchmarkTask(
                    dataset_name="tiny_linear",
                    model_name="gpr",
                    do_compile=False,
                    do_optimise=False,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
                BenchmarkTask(
                    dataset_name="tiny_linear",
                    model_name="svgp",
                    do_compile=False,
                    do_optimise=False,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
            ],
        ),
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                ],
                models=[
                    md.gpr,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False, True],
                do_optimise=[False],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            [
                BenchmarkTask(
                    dataset_name="tiny_linear",
                    model_name="gpr",
                    do_compile=False,
                    do_optimise=False,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
                BenchmarkTask(
                    dataset_name="tiny_linear",
                    model_name="gpr",
                    do_compile=True,
                    do_optimise=False,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
            ],
        ),
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                ],
                models=[
                    md.gpr,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False],
                do_optimise=[False, True],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            [
                BenchmarkTask(
                    dataset_name="tiny_linear",
                    model_name="gpr",
                    do_compile=False,
                    do_optimise=False,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
                BenchmarkTask(
                    dataset_name="tiny_linear",
                    model_name="gpr",
                    do_compile=False,
                    do_optimise=True,
                    do_predict=True,
                    do_posterior=True,
                    repetitions=2,
                ),
            ],
        ),
    ],
)
def test_benchmark_set__get_tasks(
    benchmark_set: BenchmarkSet, expected_tasks: Collection[BenchmarkTask]
) -> None:
    assert expected_tasks == benchmark_set.get_tasks()


@pytest.mark.parametrize(
    "benchmark_set,expected_metrics",
    [
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                    ds.tiny_sine,
                ],
                models=[
                    md.gpr,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False],
                do_optimise=[False],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            pd.DataFrame(
                [
                    (0, "tiny_linear", "gpr", False, False, 0),
                    (1, "tiny_sine", "gpr", False, False, 0),
                    (5, "tiny_linear", "gpr", False, False, 1),
                ],
                columns=["id", "dataset", "model", "do_compile", "do_optimise", "repetition"],
            ),
        ),
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                ],
                models=[
                    md.gpr,
                    md.svgp,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False],
                do_optimise=[False],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            pd.DataFrame(
                [
                    (0, "tiny_linear", "gpr", False, False, 0),
                    (2, "tiny_linear", "svgp", False, False, 0),
                    (5, "tiny_linear", "gpr", False, False, 1),
                ],
                columns=["id", "dataset", "model", "do_compile", "do_optimise", "repetition"],
            ),
        ),
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                ],
                models=[
                    md.gpr,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False, True],
                do_optimise=[False],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            pd.DataFrame(
                [
                    (0, "tiny_linear", "gpr", False, False, 0),
                    (3, "tiny_linear", "gpr", True, False, 0),
                    (5, "tiny_linear", "gpr", False, False, 1),
                ],
                columns=["id", "dataset", "model", "do_compile", "do_optimise", "repetition"],
            ),
        ),
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                ],
                models=[
                    md.gpr,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False],
                do_optimise=[False, True],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=2,
            ),
            pd.DataFrame(
                [
                    (0, "tiny_linear", "gpr", False, False, 0),
                    (4, "tiny_linear", "gpr", False, True, 0),
                    (5, "tiny_linear", "gpr", False, False, 1),
                ],
                columns=["id", "dataset", "model", "do_compile", "do_optimise", "repetition"],
            ),
        ),
        (
            BenchmarkSet(
                name="metrics",
                datasets=[
                    ds.tiny_linear,
                ],
                models=[
                    md.gpr,
                ],
                plots=[
                    pl.metrics_box_plot,
                ],
                do_compile=[False],
                do_optimise=[False],
                do_predict=True,
                do_posterior=True,
                file_by=GroupingSpec((GK.DATASET,), minimise=False),
                row_by=GroupingSpec((GK.METRIC,), minimise=False),
                column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
                line_by=None,
                repetitions=5,
            ),
            pd.DataFrame(
                [
                    (0, "tiny_linear", "gpr", False, False, 0),
                    (5, "tiny_linear", "gpr", False, False, 1),
                    (6, "tiny_linear", "gpr", False, False, 2),
                    (7, "tiny_linear", "gpr", False, False, 3),
                ],
                columns=["id", "dataset", "model", "do_compile", "do_optimise", "repetition"],
            ),
        ),
    ],
)
def test_benchmark_set__filter_metrics(
    benchmark_set: BenchmarkSet,
    expected_metrics: pd.DataFrame,
) -> None:
    metrics = pd.DataFrame(
        [
            (0, "tiny_linear", "gpr", False, False, 0),
            (1, "tiny_sine", "gpr", False, False, 0),
            (2, "tiny_linear", "svgp", False, False, 0),
            (3, "tiny_linear", "gpr", True, False, 0),
            (4, "tiny_linear", "gpr", False, True, 0),
            (5, "tiny_linear", "gpr", False, False, 1),
            (6, "tiny_linear", "gpr", False, False, 2),
            (7, "tiny_linear", "gpr", False, False, 3),
        ],
        columns=["id", "dataset", "model", "do_compile", "do_optimise", "repetition"],
    )
    after = benchmark_set.filter_metrics(metrics).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_metrics, after)


def test_benchmark_suite__get_tasks() -> None:
    set1 = Mock(BenchmarkSet)
    set1.name = "set1"
    set1.get_tasks.return_value = [
        BenchmarkTask(
            dataset_name="ds",
            model_name="md",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=False,
            repetitions=2,
        ),
        BenchmarkTask(
            dataset_name="ds1",
            model_name="md1",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=True,
            repetitions=2,
        ),
    ]
    set2 = Mock(BenchmarkSet)
    set2.name = "set2"
    set2.get_tasks.return_value = [
        BenchmarkTask(
            dataset_name="ds",
            model_name="md",
            do_compile=False,
            do_optimise=False,
            do_predict=False,
            do_posterior=True,
            repetitions=2,
        ),
        BenchmarkTask(
            dataset_name="ds2",
            model_name="md1",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=True,
            repetitions=2,
        ),
    ]
    set3 = Mock(BenchmarkSet)
    set3.name = "set3"
    set3.get_tasks.return_value = [
        BenchmarkTask(
            dataset_name="ds",
            model_name="md",
            do_compile=False,
            do_optimise=False,
            do_predict=False,
            do_posterior=False,
            repetitions=5,
        ),
        BenchmarkTask(
            dataset_name="ds1",
            model_name="md3",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=True,
            repetitions=2,
        ),
    ]
    suite = BenchmarkSuite(
        name="test", description="Suite used in a test.", sets=[set1, set2, set3]
    )

    assert [
        BenchmarkTask(
            dataset_name="ds",
            model_name="md",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=True,
            repetitions=5,
        ),
        BenchmarkTask(
            dataset_name="ds1",
            model_name="md1",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=True,
            repetitions=2,
        ),
        BenchmarkTask(
            dataset_name="ds2",
            model_name="md1",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=True,
            repetitions=2,
        ),
        BenchmarkTask(
            dataset_name="ds1",
            model_name="md3",
            do_compile=False,
            do_optimise=False,
            do_predict=True,
            do_posterior=True,
            repetitions=2,
        ),
    ] == suite.get_tasks()
