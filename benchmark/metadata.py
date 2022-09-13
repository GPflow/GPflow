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
Code for determining metadata for a run.
"""
import getpass
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from platform import python_version
from socket import gethostname
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import psutil
import tensorflow as tf
from cpuinfo import get_cpu_info
from git.exc import InvalidGitRepositoryError
from git.repo import Repo
from tensorflow.python.client import device_lib  # pylint: disable=no-name-in-module

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S.%f"


@dataclass(frozen=True)
class BenchmarkMetadata:
    """
    Struct with metadata about a script execution.
    """

    suite_name: str
    argv: Sequence[str]
    user: str
    hostname: str
    timestamp: str
    py_ver: str
    tf_ver: str
    np_ver: str
    ram: int
    cpu_name: str
    cpu_count: int
    cpu_frequency: int
    gpu_names: Sequence[str]
    git_branch_name: str
    git_commit: str
    run_id: str

    @staticmethod
    def create(suite_name: str) -> "BenchmarkMetadata":
        """
        Collects information about the current environment and creates a new `BenchmarkMetadata`
        with it.
        """
        suite_name = suite_name
        argv = sys.argv
        script_name = Path(argv[0]).stem
        user = getpass.getuser()
        hostname = gethostname()
        timestamp = _get_timestamp()
        py_ver = python_version()
        tf_ver = tf.__version__
        np_ver = np.__version__
        ram = psutil.virtual_memory().total
        cpu_name, cpu_count, cpu_frequency = _get_cpu_name_count_frequency()
        gpu_names = _get_gpu_names()
        git_branch_name, git_commit = _get_git_branch_and_commit()
        run_id = f"{script_name}_{suite_name}_{git_branch_name}_{timestamp}".replace("/", "_")
        return BenchmarkMetadata(
            suite_name=suite_name,
            argv=argv,
            user=user,
            hostname=hostname,
            timestamp=timestamp,
            py_ver=py_ver,
            tf_ver=tf_ver,
            np_ver=np_ver,
            ram=ram,
            cpu_name=cpu_name,
            cpu_count=cpu_count,
            cpu_frequency=cpu_frequency,
            gpu_names=gpu_names,
            git_branch_name=git_branch_name,
            git_commit=git_commit,
            run_id=run_id,
        )

    def for_shard(self) -> "BenchmarkMetadata":
        """
        Update this metadata with current machine information.

        Used in sharding, to create a machine-specific metadata, while retaining the process
        timestamp and id.
        """
        new_metadata = BenchmarkMetadata.create(self.suite_name)
        kwargs = asdict(new_metadata)
        keep_new_fields = [
            "argv",
            "user",
            "hostname",
            "ram",
            "cpu_name",
            "cpu_count",
            "cpu_frequency",
            "gpu_names",
        ]
        keep_old_fields = ["timestamp", "run_id"]
        must_match_fields = [
            "suite_name",
            "py_ver",
            "tf_ver",
            "np_ver",
            "git_branch_name",
            "git_commit",
        ]
        for field in keep_old_fields:
            del kwargs[field]
        for field in must_match_fields:
            assert getattr(self, field) == kwargs[field], (
                f"Field {field} must match between new and old metadata."
                f" Found {getattr(self, field)} and {kwargs[field]}"
            )
            del kwargs[field]
        assert keep_new_fields == list(kwargs)

        return replace(self, **kwargs)

    @property
    def gpu_name(self) -> str:
        if not self.gpu_names:
            return "No GPU"
        return " + ".join(self.gpu_names)

    def to_json(self) -> Mapping[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "BenchmarkMetadata":
        # This only works as long as our field types are sufficiently simple. We may need to update
        # this in the future:
        return BenchmarkMetadata(**data)


def _get_timestamp() -> str:
    """ Get the current date/time. """
    return datetime.utcnow().strftime(TIMESTAMP_FORMAT)


def parse_timestamp(timestamp: str) -> datetime:
    """ Parse a timestamp, as formatted in :class:`BenchmarkMetadata` into a ``datetime``. """
    return datetime.strptime(timestamp, TIMESTAMP_FORMAT)


def _get_cpu_name_count_frequency() -> Tuple[str, int, int]:
    """ Get the name, number and clock frequency of the available CPU. """
    cpu_info = get_cpu_info()
    frequency, what = cpu_info["hz_actual"]
    assert what == 0, "I have no idea what this is, but it seem to always be 0..."
    return cpu_info["brand_raw"], cpu_info["count"], frequency


def _get_gpu_names() -> Sequence[str]:
    """ Get a list of GPUs that are available to TensorFlow. """
    result = []
    for device in device_lib.list_local_devices():
        if device.device_type != "GPU":
            continue
        desc = device.physical_device_desc

        fields = desc.split(",")
        for field in fields:
            name, value = field.split(":", maxsplit=1)
            name = name.strip()
            value = value.strip()
            if name == "name":
                result.append(value)
    return result


def _get_git_branch_and_commit() -> Tuple[str, str]:
    """ Get a `(branch_name, commit_id)` tuple from the current direcory. """
    branch_name = "NO_BRANCH"
    commit = "NO_COMMIT"
    try:
        repo = Repo(__file__, search_parent_directories=True)
        try:
            branch_name = str(repo.active_branch)
        except TypeError:
            pass  # Keep current/default branch_name
        commit = str(repo.commit())
        if repo.is_dirty():
            commit += " + uncomitted changes"
    except InvalidGitRepositoryError:
        pass  # Keep current/default branch_name and commit
    return branch_name, commit
