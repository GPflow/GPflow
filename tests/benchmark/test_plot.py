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
from pathlib import Path

from benchmark.plot import _crawl_sources


def test_crawl_sources(tmp_path: Path) -> None:
    def setup_source(source_path: Path, test_id: int) -> None:
        source_path.mkdir(parents=True)
        (source_path / "metrics.csv").write_text(
            f"""test_col_1, test_col_2
{test_id}1, {test_id}2
{test_id}3, {test_id}4
"""
        )
        (source_path / "metadata.json").write_text(
            f"""{{
    "suite_name": "test",
    "argv": [],
    "user": "test_user",
    "hostname": "test_host",
    "timestamp": "20220808_085348.161698",
    "py_ver": "3.10.0",
    "tf_ver": "2.8.0",
    "np_ver": "1.20.0",
    "ram": 12,
    "cpu_name": "Test CPU",
    "cpu_count": 8,
    "cpu_frequency": 14,
    "gpu_names": ["Test GPU"],
    "git_branch_name": "develop",
    "git_commit": "1234",
    "run_id": "test_id_{test_id}"
}}
"""
        )

    source1 = tmp_path / "source1"
    source2 = tmp_path / "source2"
    source21 = source2 / "source21"
    source22 = source2 / "source22"

    setup_source(source1, 1)
    setup_source(source21, 21)
    setup_source(source22, 22)

    cycle = source22 / "cycle"
    cycle.symlink_to("../..")

    metrics_df, metadata = _crawl_sources([source1, source2])
    assert len(metrics_df) == 6
    assert {"test_id_1", "test_id_21", "test_id_22"} == set(m.run_id for m in metadata)
