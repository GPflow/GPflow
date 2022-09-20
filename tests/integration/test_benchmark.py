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
from _pytest.capture import CaptureFixture
from PIL import Image

from benchmark.benchmark_api import BENCHMARK_SUITES
from benchmark.run import run
from benchmark.sharding import ShardingStrategy


def test_benchmark(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    suite = BENCHMARK_SUITES.get("integration_test")
    dest = tmp_path / "dest"
    yes_subdir = False
    no_subdir = True
    cache_dir = tmp_path / "cache_dir"
    plot = True

    run(ShardingStrategy("start"), suite, dest, yes_subdir, cache_dir, plot)
    out_dir_str = capsys.readouterr().out.strip()
    out_dir = Path(out_dir_str)

    with open(out_dir / "metadata.json", "rt") as f:
        assert isinstance(json.load(f), dict)

    run(ShardingStrategy("0/3"), suite, out_dir, no_subdir, cache_dir, plot)
    with open(out_dir / "metadata_0of3.json", "rt") as f:
        assert isinstance(json.load(f), dict)
    assert isinstance(pd.read_csv(out_dir / "metrics_0of3.csv"), pd.DataFrame)

    run(ShardingStrategy("1/3"), suite, out_dir, no_subdir, cache_dir, plot)
    with open(out_dir / "metadata_1of3.json", "rt") as f:
        assert isinstance(json.load(f), dict)
    assert isinstance(pd.read_csv(out_dir / "metrics_1of3.csv"), pd.DataFrame)

    run(ShardingStrategy("2/3"), suite, out_dir, no_subdir, cache_dir, plot)
    with open(out_dir / "metadata_2of3.json", "rt") as f:
        assert isinstance(json.load(f), dict)
    assert isinstance(pd.read_csv(out_dir / "metrics_2of3.csv"), pd.DataFrame)

    run(ShardingStrategy("collect"), suite, out_dir, no_subdir, cache_dir, plot)
    assert isinstance(pd.read_csv(out_dir / "metrics.csv"), pd.DataFrame)
    assert isinstance(Image.open(out_dir / "metrics_tiny_linear.png"), Image.Image)
    assert isinstance(Image.open(out_dir / "metrics_tiny_sine.png"), Image.Image)
