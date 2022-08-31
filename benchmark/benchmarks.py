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
Definitions of benchmark suites.

A benchmark suite defines which datasets and models to use; and how to plot the results.
"""
import benchmark.datasets as ds
import benchmark.models as md
import benchmark.plotters as pl
from benchmark.benchmark_api import BenchmarkSet, make_benchmark_suite
from benchmark.dataset_api import DATASET_FACTORIES
from benchmark.grouping import GroupingKey as GK
from benchmark.grouping import GroupingSpec
from benchmark.model_api import MODEL_FACTORIES
from benchmark.plotter_api import PLOTTERS

make_benchmark_suite(
    name="integration_test",
    description="Suite used in tests/integration/test_benchmark.py.",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=[
                ds.tiny_linear,
                ds.tiny_sine,
            ],
            models=[
                md.gpr,
                md.svgp,
            ],
            plots=[
                pl.metrics_box_plot,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=True,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=2,
        )
    ],
)


make_benchmark_suite(
    name="ci",
    description="Suite that is run in our CI pipeline, to monitor long-term performance.",
    sets=[
        BenchmarkSet(
            name="timeline",
            datasets=[
                ds.boston,
            ],
            models=[
                md.gpr,
                md.sgpr,
                md.vgp,
                md.svgp,
            ],
            plots=[
                pl.time_line,
            ],
            do_compile=[True],
            do_optimise=[True],
            do_predict=True,
            do_posterior=True,
            file_by=GroupingSpec((GK.DATASET, GK.PLOTTER), minimise=True),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((), minimise=False),
            line_by=GroupingSpec((GK.MODEL,), minimise=False),
            repetitions=5,
        )
    ],
)


make_benchmark_suite(
    name="full",
    description="Suite that runs everything.",
    sets=[
        BenchmarkSet(
            name="metrics",
            datasets=DATASET_FACTORIES.all(),
            models=MODEL_FACTORIES.all(),
            plots=PLOTTERS.all(),
            do_compile=[False, True],
            do_optimise=[False, True],
            do_predict=True,
            do_posterior=True,
            file_by=GroupingSpec((GK.DATASET,), minimise=False),
            row_by=GroupingSpec((GK.METRIC,), minimise=False),
            column_by=GroupingSpec((GK.PLOTTER,), minimise=False),
            line_by=None,
            repetitions=5,
        )
    ],
)
