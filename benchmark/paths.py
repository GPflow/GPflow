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
Utilities for dealing with paths.
"""
from pathlib import Path

from benchmark.metadata import BenchmarkMetadata


def setup_dest(metadata: BenchmarkMetadata, dest: Path, no_subdir: bool) -> Path:
    """
    Set up destination/output directory.

    :param metadata: Data about this run.
    :param dest: Root destination directory.
    :param no_subdir: Whether to NOT create a subdirectory of ``dest``.
    :return: Path to new destination directory.
    """
    if not no_subdir:
        root = dest
        root.mkdir(parents=True, exist_ok=True)

        relative_dest = Path(metadata.run_id)
        dest = root / relative_dest
        dest.mkdir()

        latest_dir = root / "latest"
        if latest_dir.is_symlink():
            latest_dir.unlink()
        latest_dir.symlink_to(relative_dest)
    else:
        dest.mkdir(parents=True)
    return dest
