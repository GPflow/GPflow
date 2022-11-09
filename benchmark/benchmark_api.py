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
Classes and other infrastructure for defining which benchmarks to run and what plots to create.

See ``benchmarks.py`` for concrete instances of these.
"""
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from benchmark.dataset_api import DatasetFactory
from benchmark.grouping import GroupingKey, GroupingSpec
from benchmark.model_api import ModelFactory
from benchmark.plotter_api import Plotter
from benchmark.registry import Named, Registry


@dataclass(unsafe_hash=True)
class BenchmarkTask:
    """
    Definition of a single benchmark to run.

    This class is an implementation detail and needs to be a simple datastructure that is easy to
    (de-)serialise.
    """

    dataset_name: str
    """ Name of datasets to benchmark against. """

    model_name: str
    """ Name of models to benchmark. """

    do_compile: bool
    """ Whether to use ``tf.function``. """

    do_optimise: bool
    """ Whether to train the models. """

    do_predict: bool
    """ Whether to benchmark ``model.predict_f``. """

    do_posterior: bool
    """ Whether to benchmark ``model.posterior()``. """

    repetitions: int
    """ Number of times to repeat benchmarks, to estimate noise. """

    @property
    def name(self) -> str:
        tokens = []

        def print_bool(name: str) -> None:
            value_str = "T" if getattr(self, name) else "F"
            tokens.append(f"{name}={value_str}")

        tokens.append(self.dataset_name)
        tokens.append(self.model_name)
        print_bool("do_compile")
        print_bool("do_optimise")
        print_bool("do_predict")
        print_bool("do_posterior")

        return "/".join(tokens)


@dataclass(frozen=True)
class BenchmarkSet:
    """
    Defines a set of plots to produce.

    This will create plots for the cartesian product of the parameters in this class.
    See :class:`BenchmarkSuite` if you want a union of parameters.
    """

    name: str
    """ Name of this benchmark set. """

    datasets: Collection[DatasetFactory]
    """ Datasets to benchmark against. """

    models: Collection[ModelFactory]
    """ Models to benchmark. """

    plots: Collection[Plotter]
    """ Plots to generate. """

    do_compile: Collection[bool]
    """ Whether to use ``tf.function``. """

    do_optimise: Collection[bool]
    """ Whether to train the models. """

    do_predict: bool
    """ Whether to benchmark ``model.predict_f``. """

    do_posterior: bool
    """ Whether to benchmark ``model.posterior()``. """

    file_by: GroupingSpec
    """ How to split plots into different ``.png`` files. """

    column_by: GroupingSpec
    """ How to split plots into different columns within their file. """

    row_by: GroupingSpec
    """ How to split plots into different rows within their file. """

    line_by: Optional[GroupingSpec]
    """
    How to split data into different lines within a plot.

    If ``None`` data will be split by all columns not used by any of the other ``GroupingSpec``\s.
    """

    repetitions: int = 1
    """ Number of times to repeat benchmarks, to estimate noise. """

    def __post_init__(self) -> None:
        def has_unique_values(values: Collection[Any]) -> bool:
            return len(values) == len(set(values))

        def assert_unique_names(attr: str) -> None:
            names = [v.name for v in getattr(self, attr)]
            assert has_unique_values(names), f"'{attr}' must have unique names. Found: {names}."

        assert_unique_names("datasets")
        assert_unique_names("models")
        assert_unique_names("plots")

        def assert_unique_values(attr: str) -> None:
            values = getattr(self, attr)
            assert has_unique_values(values), f"'{attr}' must have unique values. Found: {values}."

        assert_unique_values("do_compile")
        assert_unique_values("do_optimise")

        def assert_disjoint_by(attr1: str, attr2: str) -> None:
            values1 = getattr(self, attr1).by
            values2 = getattr(self, attr2).by
            assert not (set(values1) & set(values2)), (
                f"'{attr1}.by' and '{attr2}.by' must be disjoint."
                f" Found: {attr1}.by={values1} and {attr2}.by={values2}."
            )

        assert_disjoint_by("file_by", "column_by")
        assert_disjoint_by("file_by", "row_by")
        assert_disjoint_by("column_by", "row_by")
        if self.line_by:
            assert_disjoint_by("file_by", "line_by")
            assert_disjoint_by("column_by", "line_by")
            assert_disjoint_by("row_by", "line_by")

        def assert_grouping_by(by: GroupingKey) -> None:
            assert (by in self.file_by.by) or (by in self.column_by.by) or (by in self.row_by.by), (
                f"Must group by '{by}' above the 'line' level. Found:"
                f" file_by={self.file_by.by},"
                f" column_by={self.column_by.by},"
                f" row_by={self.row_by.by}."
            )

        assert_grouping_by(GroupingKey.METRIC)
        assert_grouping_by(GroupingKey.PLOTTER)

    @property
    def safe_line_by(self) -> GroupingSpec:
        """
        Get ``line_by``, or a default value if ``line_by`` is ``None``.
        """
        if self.line_by is None:
            used_by: Set[GroupingKey] = (
                set(self.file_by.by) | set(self.column_by.by) | set(self.row_by.by)
            )
            line_by = set(GroupingKey) - used_by
            sorted_line_by: Sequence[GroupingKey] = sorted(line_by, key=lambda k: k.key_cost)  # type: ignore[arg-type] # for lambda
            return GroupingSpec(sorted_line_by, minimise=True)

        return self.line_by

    def get_tasks(self) -> Collection[BenchmarkTask]:
        """
        Compute ``BenchmarkTask`` objects for the cartesian product of the parameters of this
        object.
        """
        result: List[BenchmarkTask] = []
        for dataset in self.datasets:
            dataset_name = dataset.name
            for model in self.models:
                if not model.dataset_req.satisfied(dataset.tags):
                    continue

                model_name = model.name
                for do_compile in self.do_compile:
                    for do_optimise in self.do_optimise:
                        result.append(
                            BenchmarkTask(
                                dataset_name,
                                model_name,
                                do_compile,
                                do_optimise,
                                self.do_predict,
                                self.do_posterior,
                                self.repetitions,
                            )
                        )
        return result

    def filter_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a dataframe for any metrics that are not relevant to this :class:`BenchmarkSet`.
        """
        dataset_names = set(d.name for d in self.datasets)
        model_names = set(m.name for m in self.models)
        return metrics_df[
            metrics_df.dataset.isin(dataset_names)
            & metrics_df.model.isin(model_names)
            & metrics_df.do_compile.isin(self.do_compile)
            & metrics_df.do_optimise.isin(self.do_optimise)
            & (metrics_df.repetition < self.repetitions)
        ]


@dataclass(frozen=True)
class BenchmarkSuite(Named):
    """
    A union of :class:`BenchmarkSet`\s.

    This is the main definition of work in this framework.

    This will intelligently merge tasks, so if a task is present in multiple benchmark sets, it is
    executed once, and the result reused.
    """

    name: str
    description: str
    sets: Collection[BenchmarkSet]

    def __post_init__(self) -> None:
        set_names = [d.name for d in self.sets]
        assert len(set_names) == len(
            set(set_names)
        ), f"Benchmark sets must have unique names. Got: {set_names}"

    def get_tasks(self) -> Collection[BenchmarkTask]:
        """
        Computes the minimal number of benchmarks to run for this suite.
        """
        result: Dict[Tuple[str, str, bool, bool], BenchmarkTask] = {}
        for benchmark_set in self.sets:
            for task in benchmark_set.get_tasks():
                key = (task.dataset_name, task.model_name, task.do_compile, task.do_optimise)
                if key in result:
                    old_task = result[key]
                    old_task.do_predict |= task.do_predict
                    old_task.do_posterior |= task.do_posterior
                    old_task.repetitions = max(old_task.repetitions, task.repetitions)
                else:
                    result[key] = task
        return list(result.values())


BENCHMARK_SUITES: Registry[BenchmarkSuite] = Registry()


def make_benchmark_suite(**kwargs: Any) -> BenchmarkSuite:
    """
    Create a new :class:`BenchmarkSuite` and register it.

    :param kwargs: Passed on to ``BenchmarkSuite.__init__``.
    """
    result = BenchmarkSuite(**kwargs)
    BENCHMARK_SUITES.add(result)
    return result
