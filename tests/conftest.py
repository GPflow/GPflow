# Copyright 2017 the GPflow authors.
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

# code modified from https://docs.pytest.org/en/latest/example/simple.html

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skipslow", action="store_true", default=False, help="skip slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipslow"):
        # --skipslowtests is given in cli, so all tests marked with
        # pytest.mark.slow will be skipped.
        skip_slow = pytest.mark.skip(reason="Run with `--skipslowtests` to run this test.")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    else:
        # run all tests
        return
