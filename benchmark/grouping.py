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
"""
Code for grouping data for plotting.
"""
import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection, Iterator, List, Sequence, Tuple, Type

import pandas as pd

from benchmark.metadata import BenchmarkMetadata
from benchmark.plotter_api import Plotter


class GroupingKeySource(str, Enum):
    """ Sources of data to group by. """

    METADATA = "metadata"
    METRICS_DF = "metrics_df"
    PLOTTER = "plotter"


class GroupingKey(Tuple[GroupingKeySource, type, bool, float], Enum):
    """ Keys we can group by. """

    USER = (GroupingKeySource.METADATA, str, False, 1.13)
    HOSTNAME = (GroupingKeySource.METADATA, str, False, 2.01)
    TIMESTAMP = (GroupingKeySource.METADATA, str, False, 3.02)
    PY_VER = (GroupingKeySource.METADATA, str, True, 1.12)
    TF_VER = (GroupingKeySource.METADATA, str, True, 1.10)
    NP_VER = (GroupingKeySource.METADATA, str, True, 1.11)
    RAM = (GroupingKeySource.METADATA, int, False, 1.15)
    CPU_NAME = (GroupingKeySource.METADATA, str, False, 1.09)
    CPU_COUNT = (GroupingKeySource.METADATA, int, True, 1.14)
    CPU_FREQUENCY = (GroupingKeySource.METADATA, int, True, 3.01)
    GPU_NAME = (GroupingKeySource.METADATA, str, False, 1.07)
    GIT_BRANCH_NAME = (GroupingKeySource.METADATA, str, False, 1.08)

    DATASET = (GroupingKeySource.METRICS_DF, str, False, 1.03)
    MODEL = (GroupingKeySource.METRICS_DF, str, False, 1.04)
    DO_COMPILE = (GroupingKeySource.METRICS_DF, bool, True, 1.06)
    DO_OPTIMISE = (GroupingKeySource.METRICS_DF, bool, True, 1.05)
    METRIC = (GroupingKeySource.METRICS_DF, str, False, 1.02)

    PLOTTER = (GroupingKeySource.PLOTTER, Plotter, False, 1.01)

    @property
    def key_name(self) -> str:
        """ Name of this key. """
        return self.name.lower()  # pylint: disable=no-member

    @property
    def key_source(self) -> GroupingKeySource:
        """ Source of data for this key. """
        return self.value[0]  # type: ignore[no-any-return]

    @property
    def key_type(self) -> Type[Any]:
        """ Type/class of the data of this key. """
        return self.value[1]  # type: ignore[no-any-return]

    @property
    def prefix_key_to_value(self) -> bool:
        """
        Whether we should append the name of the key to the value of the key before printing.

        Some key values, such as dataset and model names, are easy to understand out of context.
        Other keys, often bools or ints, need some context to make sense. This property indicates
        whether values of this key need context.
        """
        return self.value[2]  # type: ignore[no-any-return]

    @property
    def key_cost(self) -> float:
        """
        How much we don't want to show this key.

        Should be positive.

        When trying to find a "minimal" group we will pick the group with smallest sum of costs.

        Keys the user is more likely to care about should have a small cost, while keys the user is
        unlikely to care about should have a larger cost.
        """
        return self.value[3]  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return self.name

    def __lt__(self, other: "GroupingKey") -> bool:  # type: ignore[override]
        return self.key_cost < other.key_cost


@dataclass(frozen=True)
class GroupingSpec:
    by: Sequence[GroupingKey]
    """ Keys to group by. """

    minimise: bool
    """ Whether to ignore redundant keys. """

    def __post_init__(self) -> None:
        assert len(self.by) == len(set(self.by)), f"'by' must have unique values. Found {self.by}."


def group(
    metrics_df: pd.DataFrame,
    plotters: Collection[Plotter],
    metadata: Collection[BenchmarkMetadata],
    spec: GroupingSpec,
) -> Sequence[Tuple[Tuple[str, ...], pd.DataFrame, Collection[Plotter]]]:
    """
    Group the given data according to the given specification.
    """
    result = _group(metrics_df, plotters, metadata, spec.by)

    if spec.minimise:
        # Terribly inefficient, but it's probably fast enough.
        for candidate_by in _iter_by(spec.by):
            candidate_result = _group(metrics_df, plotters, metadata, candidate_by)
            if len(result) == len(candidate_result):
                return candidate_result

    return result


def _group(
    metrics_df: pd.DataFrame,
    plotters: Collection[Plotter],
    metadata: Collection[BenchmarkMetadata],
    by: Sequence[GroupingKey],
) -> Sequence[Tuple[Tuple[str, ...], pd.DataFrame, Collection[Plotter]]]:
    """
    Group the given data by the given columns.
    """
    if not by:
        return [((), metrics_df, plotters)]

    joined_df = metrics_df

    metadata_by_columns = [k.key_name for k in by if k.key_source == GroupingKeySource.METADATA]
    if metadata_by_columns:
        metadata_columns = ["run_id"] + metadata_by_columns
        metadata_df = pd.DataFrame(
            [[getattr(md, c) for c in metadata_columns] for md in metadata],
            columns=metadata_columns,
        )
        joined_df = pd.merge(joined_df, metadata_df, on="run_id")

    plotter_by_columns = [k.key_name for k in by if k.key_source == GroupingKeySource.PLOTTER]
    if plotter_by_columns:
        plotter_key_name = GroupingKey.PLOTTER.key_name
        assert plotter_by_columns == [plotter_key_name]
        plotter_df = pd.DataFrame({plotter_key_name: [p.name for p in plotters]})
        joined_df = joined_df.merge(plotter_df, how="cross")

    result = []
    for keys, df in joined_df.groupby([k.key_name for k in by]):
        keys, key_plotters = _sanitise_keys(by, keys, plotters)
        if metadata_by_columns:
            df = df.drop(columns=metadata_by_columns)
        if plotter_by_columns:
            df = df.drop(columns=plotter_by_columns)
        result.append((keys, df, key_plotters))
    return result


def _sanitise_keys(
    by: Sequence[GroupingKey], keys: Any, plotters: Collection[Plotter]
) -> Tuple[Tuple[str, ...], Collection[Plotter]]:
    """
    Sanitises keys from Pandas group_by.

    Takes the keys from Pandas group_by, which can have all sorts of types, and:
    * Separates the plotters from the "regular" keys.
    * Converts the "regular" keys to strings.
    """
    result_keys: List[str] = []
    result_plotters = plotters
    if not isinstance(keys, tuple):
        keys = (keys,)

    for b, k in zip(by, keys):
        if b.key_type == Plotter:
            assert result_plotters == plotters
            result_plotters = tuple(p for p in plotters if p.name == k)
        else:
            key_repr = k if b.key_type == str else repr(k)
            if b.prefix_key_to_value:
                key_repr = f"{b.key_name}={key_repr}"
            result_keys.append(key_repr)

    return tuple(result_keys), result_plotters


def _iter_by(all_by: Sequence[GroupingKey]) -> Iterator[Sequence[GroupingKey]]:
    """
    Iterate over all possible subsets of grouping keys, sorted by cost.
    """
    todo: List[Tuple[float, int, List[GroupingKey]]] = [(0.0, 0, [])]
    while todo:
        _, prev_begin, prev_by = heapq.heappop(todo)
        yield prev_by
        for i, next_key in enumerate(all_by[prev_begin:]):
            next_begin = prev_begin + i + 1
            next_by = prev_by + [next_key]
            next_cost = sum(k.key_cost for k in next_by)
            heapq.heappush(todo, (next_cost, next_begin, next_by))
