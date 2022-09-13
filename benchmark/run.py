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
Main script for running benchmarks.

Usage::

    python -m benchmark.run <benchmark_suite> <output_directory>

Runs all benchmarks defined in the given suite, and plots the results. If you want to plot results
from across multiple runs, use the ``plot.py`` script.
"""
import argparse
import json
import multiprocessing as mp
import traceback
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

import benchmark.benchmarks  # pylint: disable=unused-import  # Make sure registry is loaded.
import benchmark.metrics as mt
import gpflow
from benchmark.benchmark_api import BENCHMARK_SUITES, BenchmarkSuite, BenchmarkTask
from benchmark.dataset_api import DATASET_FACTORIES
from benchmark.metadata import BenchmarkMetadata
from benchmark.metric_api import Metric
from benchmark.model_api import MODEL_FACTORIES
from benchmark.paths import setup_dest
from benchmark.plot import plot as plot_
from benchmark.sharding import ShardingStrategy

BASE_SEED = 20220721
ONE_HOUR_S = 60 * 60


def _collect_metrics(
    task: BenchmarkTask,
    data_cache_dir: Path,
    random_seed: int,
) -> Optional[Mapping[Metric, float]]:
    print("Loading dataset:", task.dataset_name)
    dataset_fac = DATASET_FACTORIES.get(task.dataset_name)
    dataset = dataset_fac.create_dataset(data_cache_dir)
    print(tabulate(dataset.stats, showindex="never"))
    train_data = dataset.train
    test_data = dataset.test

    rng = np.random.default_rng(random_seed)
    model_fac = MODEL_FACTORIES.get(task.model_name)
    model = model_fac.create_model(train_data, rng)

    metrics = {}

    model.predict_y(test_data.X)  # Warm-up TF.

    print("Model before training:")
    gpflow.utilities.print_summary(model)

    if task.do_optimise:
        t_before = perf_counter()
        loss_fn = gpflow.models.training_loss_closure(model, train_data.XY, compile=task.do_compile)
        opt_log = gpflow.optimizers.Scipy().minimize(
            loss_fn,
            variables=model.trainable_variables,
            compile=task.do_compile,
            options={"disp": 10, "maxiter": 1_000},
        )
        t_after = perf_counter()
        n_iter = opt_log.nit
        t_train = t_after - t_before
        metrics[mt.n_training_iterations] = n_iter
        metrics[mt.training_time] = t_train
        metrics[mt.training_iteration_time] = t_train / n_iter
        print(f"Training took {t_after - t_before}s for {n_iter} iterations.")
        print("Model after training:")
        gpflow.utilities.print_summary(model)

    likelihood = model.likelihood

    if task.do_predict:
        t_before = perf_counter()
        f_m, f_v = model.predict_f(test_data.X)
        t_after = perf_counter()
        metrics[mt.prediction_time] = t_after - t_before

        metrics[mt.nlpd] = -np.sum(
            likelihood.predict_log_density(test_data.X, f_m, f_v, test_data.Y)
        )

        y_m, _y_v = likelihood.predict_mean_and_var(test_data.X, f_m, f_v)
        error = test_data.Y - y_m
        metrics[mt.mae] = np.average(np.abs(error))
        metrics[mt.rmse] = np.average(error ** 2) ** 0.5

    if task.do_posterior:
        t_before = perf_counter()
        posterior = model.posterior()
        posterior.predict_f(np.zeros((0, test_data.D)))
        t_after = perf_counter()
        metrics[mt.posterior_build_time] = t_after - t_before

        t_before = perf_counter()
        f_m, f_v = posterior.predict_f(test_data.X)
        t_after = perf_counter()
        metrics[mt.posterior_prediction_time] = t_after - t_before

        metrics[mt.posterior_nlpd] = -np.sum(
            likelihood.predict_log_density(test_data.X, f_m, f_v, test_data.Y)
        )

        y_m, _y_v = likelihood.predict_mean_and_var(test_data.X, f_m, f_v)
        error = test_data.Y - y_m
        metrics[mt.posterior_mae] = np.average(np.abs(error))
        metrics[mt.posterior_rmse] = np.average(error ** 2) ** 0.5

    metrics_values = list(metrics.values())

    # pylint: disable=no-member
    assert np.isfinite(metrics_values).all(), f"{metrics} not all finite."

    return metrics


def _collect_metrics_process(
    task: BenchmarkTask,
    data_cache_dir: Path,
    random_seed: int,
    metrics_queue: "mp.Queue[Optional[Mapping[Metric, float]]]",
) -> None:
    metrics: Optional[Mapping[Metric, float]] = None
    try:
        metrics = _collect_metrics(task, data_cache_dir, random_seed)
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
    metrics_queue.put(metrics)


def _run_benchmarks(
    metadata: BenchmarkMetadata,
    shard: ShardingStrategy,
    suite: BenchmarkSuite,
    dest: Path,
    cache_dir: Path,
) -> Optional[pd.DataFrame]:
    data_cache_dir = cache_dir / "data"

    # Run all tasks in separate processes, to make sure all memory etc. is released.
    mp_ctx = mp.get_context("spawn")

    tasks = suite.get_tasks()
    global_repetition = 0
    total_repetitions = sum(t.repetitions for t in tasks)
    max_task_repetitions = max(t.repetitions for t in tasks)

    metrics_list = []
    metrics_df: Optional[pd.DataFrame] = None

    t_before = perf_counter()
    for task_repetition in range(max_task_repetitions):
        for task in tasks:
            if task_repetition >= task.repetitions:
                continue

            if global_repetition % shard.shard_n != shard.shard_i:
                global_repetition += 1
                continue

            for retry in range(3):
                t_now = perf_counter()
                fraction_done = global_repetition / total_repetitions
                if global_repetition == 0:
                    eta_str = "???"
                else:
                    velocity = (t_now - t_before) / global_repetition
                    t_end = velocity * total_repetitions + t_before
                    t_remaining = t_end - t_now
                    eta_str = f"{timedelta(seconds=t_remaining)}"
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(f"Running: {task.name}/rep={task_repetition} (retry {retry})")
                print(f"{fraction_done: .2%} done. ETA: {eta_str}.")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

                random_seed = BASE_SEED + task_repetition + retry * task.repetitions
                metrics_queue: "mp.Queue[Optional[Mapping[Metric, float]]]" = mp_ctx.Queue()
                metrics_process = mp_ctx.Process(
                    target=_collect_metrics_process,
                    args=(task, data_cache_dir, random_seed, metrics_queue),
                )
                metrics_process.start()
                metrics = metrics_queue.get(timeout=ONE_HOUR_S)
                metrics_process.join()
                assert metrics_process.exitcode == 0
                if metrics is None:
                    print("Model failed. Retrying with another random seed.")
                    continue

                for metric, metric_value in metrics.items():
                    metrics_list.append(
                        (
                            metadata.run_id,
                            task.dataset_name,
                            task.model_name,
                            task.do_compile,
                            task.do_optimise,
                            task_repetition,
                            metric.name,
                            metric_value,
                        )
                    )
                metrics_df = pd.DataFrame(
                    metrics_list,
                    columns=[
                        "run_id",
                        "dataset",
                        "model",
                        "do_compile",
                        "do_optimise",
                        "repetition",
                        "metric",
                        "value",
                    ],
                )
                print(tabulate(metrics_df, headers="keys", showindex="never"))
                metrics_path = dest / f"metrics{shard.file_suffix}.csv"
                metrics_df.to_csv(metrics_path, index=False)
                break
            global_repetition += 1

    return metrics_df


def run(
    shard: ShardingStrategy,
    suite: BenchmarkSuite,
    dest: Path,
    no_subdir: bool,
    cache_dir: Path,
    plot: bool,
) -> None:
    """ Collect data for a benchmark suite. """
    if shard.setup_dest:
        metadata = BenchmarkMetadata.create(suite.name)
        dest = setup_dest(metadata, dest, no_subdir)
    else:
        assert no_subdir, "Must use 'no-subdir=True' with 'shard={shard}'."
        with open(dest / "metadata.json", "rt") as f:
            metadata = BenchmarkMetadata.from_json(json.load(f))
        assert metadata.suite_name == suite.name
        metadata = metadata.for_shard()

    if shard.print_dest:
        print(dest.resolve())

    if shard.write_metadata:
        metadata_path = dest / f"metadata{shard.file_suffix}.json"
        with open(metadata_path, "wt") as f:
            json.dump(metadata.to_json(), f, indent=4)

    metrics_df: Optional[pd.DataFrame] = None

    if shard.run_benchmarks:
        metrics_df = _run_benchmarks(
            metadata,
            shard,
            suite,
            dest,
            cache_dir,
        )

    if shard.collect:
        assert metrics_df is None
        metrics_path = dest / "metrics.csv"
        metrics_df_list = [pd.read_csv(p) for p in shard.find_shards(metrics_path)]
        metrics_df = pd.concat(metrics_df_list, axis="index", ignore_index=True)
        metrics_df.to_csv(metrics_path, index=False)

    if shard.plot and plot and (metrics_df is not None):
        plot_(
            metadata=metadata,
            suite=suite,
            results_df=metrics_df,
            results_metadata=[metadata],
            dest=dest,
        )


def run_from_argv() -> None:
    """ Collect data for a benchmark suite, based on command line arguments. """
    parser = argparse.ArgumentParser(description="Run a benchmark suite.")
    parser.add_argument(
        "--no_subdir",
        "--no-subdir",
        action="store_true",
        help="By default this script will create a subdirectory inside `dest` and write output to"
        " that. Setting this flag will suppress the subdirectory and use `dest` directly.",
    )
    parser.add_argument(
        "--no_plot",
        "--no-plot",
        action="store_true",
        help="By default this script will create plots of data."
        " Setting this flag will suppress plotting.",
    )
    parser.add_argument(
        "--cache_dir",
        "--cache-dir",
        type=Path,
        default=Path("/tmp/gpflow_benchmark_data"),
        help="Where to cache stuff, like datasets.",
    )
    parser.add_argument(
        "--shard",
        default=ShardingStrategy("no"),
        type=ShardingStrategy,
        help=(
            "Sharding strategy:"
            " If set to 'no' this script performs all necessary work."
            " If set to 'start' this script creates an output directory for the shards, and"
            " prints its path."
            " If set to the format '<i>/<n>', where 0 <= i < n then this script only runs"
            " benchmarks for shard 'i' out of 'n', and stores partial results in 'dest'."
            " This requires '--no_subdir'."
            " If set to 'collect' then this script assumes all benchmarks already have been"
            " computed, using the '<i>/<n>' commands, and merges their results into one file."
            " This requires '--no_subdir'."
        ),
    )
    parser.add_argument(
        "suite",
        type=str,
        choices=BENCHMARK_SUITES.names(),
        help="Which benchmark suite to run.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Directory to write results to.",
    )

    args = parser.parse_args()
    suite = BENCHMARK_SUITES.get(args.suite)
    run(
        args.shard,
        suite,
        args.dest,
        args.no_subdir,
        args.cache_dir,
        not args.no_plot,
    )


if __name__ == "__main__":
    run_from_argv()
