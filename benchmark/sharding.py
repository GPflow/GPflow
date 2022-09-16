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
Code for managing sharding of benchmarks.

This will only create one metadata for the entire sharded run, and assume that all shards are run on
similar machines.
"""
from pathlib import Path
from typing import Collection


class ShardingStrategy:
    """
    Strategy for how to shard (split) the work.
    """

    def __init__(self, spec: str) -> None:
        """
        Valid ``spec``\s are:

        - ``no``: this script performs all necessary work.
        - ``start``: this script creates an output directory for the shards, and
          prints its path."
        - ``<i>/<n>``:, where 0 <= i < n this script only runs benchmarks for shard ``i`` out of
          ``n``, and stores partial results.
        - ``collect``: this script assumes all benchmarks already have been computed, using the
          ``<i>/<n>`` commands, and merges their results into one file.
        """
        self.spec = spec
        if spec == "no":
            self.setup_dest = True
            self.print_dest = False
            self.write_metadata = True
            self.run_benchmarks = True
            self.shard_i = 0
            self.shard_n = 1
            self.add_file_suffix = False
            self.collect = False
            self.plot = True
        elif spec == "start":
            self.setup_dest = True
            self.print_dest = True
            self.write_metadata = True
            self.run_benchmarks = False
            self.shard_i = 0
            self.shard_n = 1
            self.add_file_suffix = False
            self.collect = False
            self.plot = False
        elif spec == "collect":
            self.setup_dest = False
            self.print_dest = False
            self.write_metadata = False
            self.run_benchmarks = False
            self.shard_i = 0
            self.shard_n = 1
            self.add_file_suffix = False
            self.collect = True
            self.plot = True
        else:
            i_str, n_str = spec.split("/")
            self.setup_dest = False
            self.print_dest = False
            self.write_metadata = True
            self.run_benchmarks = True
            self.shard_i = int(i_str)
            self.shard_n = int(n_str)
            self.add_file_suffix = True
            self.collect = False
            self.plot = False
        assert not (self.run_benchmarks and self.collect)
        assert 0 <= self.shard_i < self.shard_n, (self.shard_i, self.shard_n)

    @property
    def file_suffix(self) -> str:
        return f"_{self.shard_i}of{self.shard_n}" if self.add_file_suffix else ""

    def find_shards(self, base_path: Path) -> Collection[Path]:
        parent = base_path.parent
        dot_suffixes = "".join(base_path.suffixes)
        glob = f"{base_path.stem}_*of*{dot_suffixes}"
        return tuple(parent.glob(glob))

    def __repr__(self) -> str:
        return self.spec
