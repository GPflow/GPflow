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
from typing import Collection, Sequence, Tuple
from unittest.mock import Mock

import pandas as pd
import pytest

from benchmark.grouping import GroupingKey as GK
from benchmark.grouping import GroupingSpec, _iter_by, group
from benchmark.metadata import BenchmarkMetadata
from benchmark.plotter_api import Plotter

METRICS_COLUMNS = ["run_id", "dataset", "model", "metric", "value"]
METRICS_DF = pd.DataFrame(
    [
        ("run_1", "ds1", "md1", "mt1", 1.0),
        ("run_2", "ds1", "md1", "mt1", 2.0),
        ("run_1", "ds2", "md1", "mt1", 3.0),
        ("run_1", "ds1", "md2", "mt1", 4.0),
        ("run_1", "ds1", "md1", "mt2", 5.0),
    ],
    columns=METRICS_COLUMNS,
)

PL1 = Mock(Plotter)
PL1.name = "pl1"
PL2 = Mock(Plotter)
PL2.name = "pl2"

BM1 = Mock(BenchmarkMetadata)
BM1.run_id = "run_1"
BM1.git_branch_name = "branch_1"
BM1.user = "user_1"
BM1.timestamp = "20220805_111111.111111"

BM2 = Mock(BenchmarkMetadata)
BM2.run_id = "run_2"
BM2.git_branch_name = "branch_2"
BM2.user = "user_2"
BM2.timestamp = "20220805_222222.222222"


def test_iter_by() -> None:
    assert [
        [],
        [GK.USER],
        [GK.RAM],
        [GK.HOSTNAME],
        [GK.USER, GK.RAM],
        [GK.TIMESTAMP],
        [GK.USER, GK.HOSTNAME],
        [GK.HOSTNAME, GK.RAM],
        [GK.USER, GK.TIMESTAMP],
        [GK.TIMESTAMP, GK.RAM],
        [GK.USER, GK.HOSTNAME, GK.RAM],
        [GK.HOSTNAME, GK.TIMESTAMP],
        [GK.USER, GK.TIMESTAMP, GK.RAM],
        [GK.USER, GK.HOSTNAME, GK.TIMESTAMP],
        [GK.HOSTNAME, GK.TIMESTAMP, GK.RAM],
        [GK.USER, GK.HOSTNAME, GK.TIMESTAMP, GK.RAM],
    ] == list(
        _iter_by(
            [
                GK.USER,
                GK.HOSTNAME,
                GK.TIMESTAMP,
                GK.RAM,
            ]
        )
    )


@pytest.mark.parametrize(
    "spec,expected",
    [
        # Group by no key:
        (
            GroupingSpec(by=[], minimise=False),
            [
                (
                    (),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1, PL2],
                ),
            ],
        ),
        # Group by a metadata key:
        (
            GroupingSpec(by=[GK.GIT_BRANCH_NAME], minimise=False),
            [
                (
                    ("branch_1",),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1, PL2],
                ),
                (
                    ("branch_2",),
                    pd.DataFrame(
                        [
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1, PL2],
                ),
            ],
        ),
        # Group by a metrics_df key:
        (
            GroupingSpec(by=[GK.DATASET], minimise=False),
            [
                (
                    ("ds1",),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1, PL2],
                ),
                (
                    ("ds2",),
                    pd.DataFrame(
                        [
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1, PL2],
                ),
            ],
        ),
        # Group by a (the) plotter key:
        (
            GroupingSpec(by=[GK.PLOTTER], minimise=False),
            [
                (
                    (),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    (),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
            ],
        ),
        # Group by all the keys:
        (
            GroupingSpec(
                by=[
                    GK.USER,
                    GK.GIT_BRANCH_NAME,
                    GK.TIMESTAMP,
                    GK.DATASET,
                    GK.MODEL,
                    GK.METRIC,
                    GK.PLOTTER,
                ],
                minimise=False,
            ),
            [
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds1", "md1", "mt2"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds1", "md1", "mt2"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds1", "md2", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds1", "md2", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds2", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("user_1", "branch_1", "20220805_111111.111111", "ds2", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("user_2", "branch_2", "20220805_222222.222222", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("user_2", "branch_2", "20220805_222222.222222", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
            ],
        ),
        # Group by all the keys, and minimise:
        (
            GroupingSpec(
                by=[
                    GK.USER,
                    GK.GIT_BRANCH_NAME,
                    GK.TIMESTAMP,
                    GK.DATASET,
                    GK.MODEL,
                    GK.METRIC,
                    GK.PLOTTER,
                ],
                minimise=True,
            ),
            [
                (
                    ("branch_1", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("branch_1", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt1", 1.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("branch_1", "ds1", "md1", "mt2"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("branch_1", "ds1", "md1", "mt2"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md1", "mt2", 5.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("branch_1", "ds1", "md2", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("branch_1", "ds1", "md2", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds1", "md2", "mt1", 4.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("branch_1", "ds2", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("branch_1", "ds2", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_1", "ds2", "md1", "mt1", 3.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
                (
                    ("branch_2", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL1],
                ),
                (
                    ("branch_2", "ds1", "md1", "mt1"),
                    pd.DataFrame(
                        [
                            ("run_2", "ds1", "md1", "mt1", 2.0),
                        ],
                        columns=METRICS_COLUMNS,
                    ),
                    [PL2],
                ),
            ],
        ),
    ],
)
def test_group(
    spec: GroupingSpec,
    expected: Sequence[Tuple[Tuple[str, ...], pd.DataFrame, Collection[Plotter]]],
) -> None:
    actual = group(METRICS_DF, [PL1, PL2], [BM1, BM2], spec)

    assert len(expected) == len(actual)
    for expected_gr, actual_gr in zip(expected, actual):
        expected_key, expected_df, expected_plotters = expected_gr
        actual_key, actual_df, actual_plotters = actual_gr

        assert expected_key == actual_key
        pd.testing.assert_frame_equal(expected_df, actual_df.reset_index(drop=True))
        assert expected_plotters == list(actual_plotters)
