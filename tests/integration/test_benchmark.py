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
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from benchmark.benchmark_api import BENCHMARK_SUITES
from benchmark.metadata import BenchmarkMetadata
from benchmark.run import run


def test_benchmark(tmp_path: Path) -> None:
    suite = BENCHMARK_SUITES.get("integration_test")
    metadata = BenchmarkMetadata.create(suite.name)
    dest = tmp_path / "dest"
    dest.mkdir(parents=True)
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir(parents=True)
    plot = True

    run(metadata, suite, dest, cache_dir, plot)

    with open(dest / "metadata.json", "rt") as f:
        assert isinstance(json.load(f), dict)
    assert isinstance(pd.read_csv(dest / "metrics.csv"), pd.DataFrame)
    assert isinstance(Image.open(dest / "metrics_tiny_linear.png"), Image.Image)
    assert isinstance(Image.open(dest / "metrics_tiny_sine.png"), Image.Image)
