#  Copyright 2021 The GPflow Contributors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import cast

import numpy as np
import pytest
from _pytest.fixtures import SubRequest


@pytest.fixture(name="full_cov", params=[True, False])
def _full_cov_fixture(request: SubRequest) -> bool:
    return cast(bool, request.param)


@pytest.fixture(name="full_output_cov", params=[True, False])
def _full_output_cov_fixture(request: SubRequest) -> bool:
    return cast(bool, request.param)


@pytest.fixture(name="whiten", params=[True, False])
def _whiten_fixture(request: SubRequest) -> bool:
    return cast(bool, request.param)


@pytest.fixture(name="rng")
def _rng_fixture() -> np.random.Generator:
    return np.random.default_rng(20220523)
