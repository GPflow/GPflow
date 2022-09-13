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
Concrete code for plotting results.
"""
from dataclasses import replace
from math import isnan
from typing import Collection, Tuple

import pandas as pd
from matplotlib.axes import Axes

from benchmark.grouping import GroupingKey, GroupingSpec, group
from benchmark.metadata import BenchmarkMetadata, parse_timestamp
from benchmark.metric_api import METRICS, Metric
from benchmark.plotter_api import make_plotter


def _join_key(key: Tuple[str, ...]) -> str:
    return "; ".join(key)


def _get_metric(metrics_df: pd.DataFrame) -> Metric:
    (metric_name,) = metrics_df.metric.unique()
    return METRICS.get(metric_name)


def _shared_ax_config(
    ax: Axes,
    file_key: Tuple[str, ...],
    column_key: Tuple[str, ...],
    row_key: Tuple[str, ...],
    metric: Metric,
) -> None:

    title_key = column_key + row_key
    if metric.name in title_key:
        title_key = tuple(k for k in title_key if k != metric.name)
    if title_key:
        ax.set_title(_join_key(title_key))

    ax.tick_params(axis="x", labelrotation=30)

    if metric.unit is None:
        y_label = metric.pretty_name
    else:
        y_label = f"{metric.pretty_name} ({metric.unit})"
    ax.set_ylabel(y_label)

    if metric.lower_bound is not None:
        ax.set_ylim(bottom=metric.lower_bound)
    if metric.upper_bound is not None:
        ax.set_ylim(top=metric.upper_bound)


@make_plotter
def metrics_box_plot(
    ax: Axes,
    file_key: Tuple[str, ...],
    column_key: Tuple[str, ...],
    row_key: Tuple[str, ...],
    line_by: GroupingSpec,
    metrics_df: pd.DataFrame,
    metadata: Collection[BenchmarkMetadata],
) -> None:
    """
    Creates box-plots of metrics.
    """
    assert GroupingKey.PLOTTER not in line_by.by
    metric = _get_metric(metrics_df)
    labels = []
    values = []
    for key, df, _ in group(metrics_df, [], metadata, line_by):
        labels.append(_join_key(key))
        values.append(df.value)
    ax.boxplot(values, labels=labels)

    _shared_ax_config(ax, file_key, column_key, row_key, metric)


@make_plotter
def time_line(
    ax: Axes,
    file_key: Tuple[str, ...],
    column_key: Tuple[str, ...],
    row_key: Tuple[str, ...],
    line_by: GroupingSpec,
    metrics_df: pd.DataFrame,
    metadata: Collection[BenchmarkMetadata],
) -> None:
    """
    Creates a plot of performance over time.
    """
    assert GroupingKey.PLOTTER not in line_by.by
    metric = _get_metric(metrics_df)
    line_by = replace(line_by, by=[k for k in line_by.by if k != GroupingKey.TIMESTAMP])
    line_groups = group(metrics_df, [], metadata, line_by)
    for key, df, _ in line_groups:
        line_xs = []
        line_y_means = []
        line_y_uppers = []
        line_y_lowers = []

        scatter_xs = []
        scatter_ys = []

        timestamp_by = GroupingSpec([GroupingKey.TIMESTAMP], minimise=False)
        timestamp_groups = group(df, [], metadata, timestamp_by)
        for (timestamp,), timestamp_df, _ in timestamp_groups:
            parsed_timestamp = parse_timestamp(timestamp)
            line_xs.append(parsed_timestamp)
            y_mean = timestamp_df.value.mean()
            y_std = timestamp_df.value.std()
            if isnan(y_std):
                y_std = 0.0
            line_y_means.append(y_mean)
            line_y_uppers.append(y_mean + 1.96 * y_std)
            line_y_lowers.append(y_mean - 1.96 * y_std)

            scatter_xs.extend(len(timestamp_df) * [parsed_timestamp])
            scatter_ys.extend(timestamp_df.value)

        (mean_line,) = ax.plot(line_xs, line_y_means, label=_join_key(key))
        color = mean_line.get_color()
        ax.fill_between(line_xs, line_y_lowers, line_y_uppers, color=color, alpha=0.3)
        ax.scatter(scatter_xs, scatter_ys, color=color)

    if len(line_groups) > 1:
        ax.legend()

    _shared_ax_config(ax, file_key, column_key, row_key, metric)
