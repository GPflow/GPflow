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
Script for plotting data created by ``run.py``.

The ``run.py`` script generally generates its own plots, but this scripet is useful if you want to
plot results from multiple runs, or if you have updated some code and want to regenerate the plots.

Usage::

    python -m benchmark.plot <benchmark_suite> <input_directories...> <output_directory>

Multiple input directories can be defined, and all of them will be recursively crawled to find all
data. This data is then plotted according to the given benchmark suite.
"""
import argparse
import json
from pathlib import Path
from typing import Collection, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import benchmark.benchmarks  # pylint: disable=unused-import  # Make sure registry is loaded.
import benchmark.metrics  # pylint: disable=unused-import  # Make sure registry is loaded.
from benchmark.benchmark_api import BENCHMARK_SUITES, BenchmarkSuite
from benchmark.grouping import group
from benchmark.metadata import BenchmarkMetadata
from benchmark.paths import setup_dest


def _crawl_sources(sources: Sequence[Path]) -> Tuple[pd.DataFrame, Collection[BenchmarkMetadata]]:
    results_df_list = []
    results_metadata = []
    visited = set()
    queue = list(sources)
    while queue:
        source = queue.pop()

        source = source.resolve()
        assert source.is_dir()
        # Sym-links can cause cycles - don't visit the same directory multiple times...
        if source in visited:
            continue
        visited.add(source)

        metrics_df_path = source / "metrics.csv"
        metadata_path = source / "metadata.json"
        if metrics_df_path.exists() or metadata_path.exists():
            assert metrics_df_path.exists()
            assert metadata_path.exists()
            results_df_list.append(pd.read_csv(metrics_df_path))
            with open(metadata_path, "rt") as f:
                results_metadata.append(BenchmarkMetadata.from_json(json.load(f)))

        for child in source.iterdir():
            if child.is_dir():
                queue.append(child)

    results_df = pd.concat(results_df_list, axis="index", ignore_index=True)
    return results_df, results_metadata


def plot(
    metadata: BenchmarkMetadata,
    suite: BenchmarkSuite,
    results_df: pd.DataFrame,
    results_metadata: Collection[BenchmarkMetadata],
    dest: Path,
) -> None:
    """ Plot previously collected metrics. """
    for benchmark_set in suite.sets:
        benchmark_set_df = benchmark_set.filter_metrics(results_df)
        for file_key, file_df, file_plotters in group(
            benchmark_set_df, benchmark_set.plots, results_metadata, benchmark_set.file_by
        ):
            row_groups = group(file_df, file_plotters, results_metadata, benchmark_set.row_by)
            n_rows = len(row_groups)
            n_columns = len(
                group(file_df, file_plotters, results_metadata, benchmark_set.column_by)
            )

            width = 6 * n_columns
            height = 4 * n_rows
            fig, axes = plt.subplots(
                ncols=n_columns,
                nrows=n_rows,
                figsize=(width, height),
                squeeze=False,
                dpi=100,
            )

            for row_axes, (row_key, row_df, row_plotters) in zip(axes, row_groups):
                column_groups = group(
                    row_df, row_plotters, results_metadata, benchmark_set.column_by
                )
                for column_ax, (column_key, column_df, column_plotters) in zip(
                    row_axes, column_groups
                ):
                    (plotter,) = column_plotters
                    plotter.plot(
                        column_ax,
                        file_key,
                        column_key,
                        row_key,
                        benchmark_set.safe_line_by,
                        column_df,
                        results_metadata,
                    )

            fig.tight_layout()
            file_tokens = (benchmark_set.name,) + file_key
            fig.savefig(dest / f"{'_'.join(file_tokens)}.png")
            plt.close(fig)


def plot_from_argv() -> None:
    """ Plot previously collected metrics, based on command line arguments. """
    parser = argparse.ArgumentParser(description="Plot a benchmark suite.")
    parser.add_argument(
        "--no_subdir",
        "--no-subdir",
        action="store_true",
        help="By default this script will create a subdirectory inside `dest` and write output to"
        " that. Setting this flag will suppress the subdirectory and use `dest` directly.",
    )
    parser.add_argument(
        "suite",
        type=str,
        choices=BENCHMARK_SUITES.names(),
        help="Which benchmark suite to run.",
    )
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="Directory(s) to read data from."
        " These will be recursively crawled to find all `metrics.csv`s.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Directory to write results to.",
    )
    args = parser.parse_args()

    metadata = BenchmarkMetadata.create(args.suite)
    suite = BENCHMARK_SUITES.get(args.suite)
    results_df, results_metadata = _crawl_sources(args.sources)
    dest = setup_dest(metadata, args.dest, args.no_subdir)
    plot(metadata, suite, results_df, results_metadata, dest)


if __name__ == "__main__":
    plot_from_argv()
