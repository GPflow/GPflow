#  Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
from inspect import isabstract
from typing import DefaultDict, Iterable, Set, Type

import pytest

import gpflow.ci_utils
from gpflow.posteriors import AbstractPosterior


@pytest.fixture(name="tested_posteriors", scope="package")
def _tested_posteriors() -> DefaultDict[str, Set[Type[AbstractPosterior]]]:
    return DefaultDict(set)


@pytest.fixture(scope="package", autouse=True)
def _ensure_all_posteriors_are_tested_fixture(
    tested_posteriors: DefaultDict[str, Set[Type[AbstractPosterior]]]
) -> Iterable[None]:
    """
    This fixture ensures that all concrete posteriors have unit tests which compare the predictions
    from the fused and precomputed code paths. When adding a new concrete posterior class to
    GPFlow, ensure that it is also tested in this manner.

    This autouse, package scoped fixture will always be executed when tests in this package are run.
    """
    # Code here will be executed before any of the tests in this package.

    yield  # Run tests in this package.

    # Code here will be executed after all of the tests in this package.

    available_posteriors = list(gpflow.ci_utils.subclasses(AbstractPosterior))
    concrete_posteriors = set([k for k in available_posteriors if not isabstract(k)])

    messages = []
    for key, key_tested_posteriors in tested_posteriors.items():
        untested_posteriors = concrete_posteriors - key_tested_posteriors
        if untested_posteriors:
            messages.append(
                f"For key '{key}' no tests have been registered for the following posteriors: {untested_posteriors}."
            )

    if messages:
        raise AssertionError("\n".join(messages))
