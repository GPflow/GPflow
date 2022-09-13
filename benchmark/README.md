# GFflow benchmarking tools

## Quick start

Make sure you install everything. This depends on `tests_requirements.txt`.

From the GPflow root directory run:

```bash
pip install -e . -r tests_requirements.txt tensorflow~=2.8.0 tensorflow-probability~=0.16.0
```

You use `run.py` to run the benchmarks. You need to tell it which "suite" to run, and where to put
the results. For example:

```bash
python -m benchmark.run ci ~/experiment_results/
```

By default `run.py` will also make plots of its results. If you want to run plotting separately you
can use the `plot.py` script. There are a couple of reasons one might want to do that. Maybe:

* You've changed the plotting code and want to regenerate plots.
* You want to compare results from different git branches.
* You want to compare results from different environments (Different Python or TF versions?
  Different hardware?)

To run `plot.py` you need to tell it which "suite" to run, where to find previous results, and where
to put new results. For example:

```bash
python -m benchmark.plot ci ~/experiment_results/ ~/experiment_results/
```

### Sharding

The `run.py` script supports running benchmarks in parallel on different machines.
For example:

```bash
out_dir="$(python -m benchmark.run --shard ci ~/experiment_results/)"

# The directory `${out_dir}` should now be distributed to the worker machines.
# These three lines can then run in parallel on the worker machines:
python -m benchmark.run --shard 0/3 --no-subdir ci ${out_dir}
python -m benchmark.run --shard 1/3 --no-subdir ci ${out_dir}
python -m benchmark.run --shard 2/3 --no-subdir ci ${out_dir}

# Finally collect and merge `$out_dir` from the worker machines, then run:
python -m benchmark.run --shard collect --no-subdir ci ${out_dir}
```

## Understanding the main concepts

### Datasets

Unsurprisingly datasets are an important concept in GPflow benchmarking.
To hold the data itself we have the `Dataset` class. Since datasets can be large we don't want to
hold them in memory for too long, so generally datasets are represented by a `DatasetFactory`, which
can load them on demand.

Datasets can be tagged by `DatasetTag`s, which indicate the properties of
the dataset.

All the dataset-related concepts are defined in `dataset_api.py`. Datasets themselves are defined in
`datasets.py`.

### Models

Another big non-surprise is that models are an important concept. For models we use the GPflow
`gpflow.models.GPModel` type. Each model need to be instantiated specifically for each dataset, so
we represent models by a `ModelFactory`, which can then create the actual model once the dataset has
been loaded into memory.

Not all models are compatible with all datasets. We use the dataset tags to define which datasets a
model is compatible with. Models can in turn also have tags, indicating the properties of the model.

All the model-related concepts are defined in `model_api.py`. Model factories themselves are defined
in `models.py`.

### Plotters

Once we have run some models on some datasets and collected some metrics we want to plot those
metrics. The `Plotter` class is responsible for turning metrics into plots.

As you can probably guess by now, the plotter infrastructure is defined in `plotter_api.py`, while
concrete plotters themselves are defined in `plotters.py`.

### Benchmarks

Benchmarks are then combinations of datasets, models and plotters.

A `BenchmarkSet` is the cartesian product of some datasets, some models and some plotters. We run
all models against all the (compatible) datasets and create all the plots.

A `BenchmarkSuite` is a collection of `BenchmarkSet`s, and these are the main definition of what
work the `run.py` and `plot.py` should do. The framework is smart enough that if multiple benchmark
sets within the same suite requires the same dataset/model combination, it will only be run once.

All the benchmark-related concepts are defined in `benchmark_api.py`. Concrete benchmark suites are
defined in `benchmarks.py`.
