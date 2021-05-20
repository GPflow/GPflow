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
import warnings
from inspect import isabstract

import pytest

import gpflow

TESTED_POSTERIORS = set()


@pytest.fixture(scope="package", autouse=True)
def _ensure_all_posteriors_are_tested_fixture():
    """
    This fixture ensures that all concrete posteriors have unit tests which compare the predictions
    from the fused and precomputed code paths. When adding a new concrete posterior class to
    GPFlow, ensure that it is also tested in this manner.

    This autouse, module scoped fixture will always be executed when tests in this module are run.
    """
    # Code here will be executed before any of the tests in this module.

    yield  # Run tests in this module.

    # Code here will be executed after all of the tests in this module.

    available_posteriors = list(gpflow.ci_utils.subclasses(gpflow.posteriors.Posterior))
    concrete_posteriors = set([k for k in available_posteriors if not isabstract(k)])

    untested_posteriors = concrete_posteriors - TESTED_POSTERIORS

    if untested_posteriors:
        message = (
            f"No tests have been registered for the following posteriors: {untested_posteriors}."
        )
        if gpflow.ci_utils.is_continuous_integration():
            raise AssertionError(message)
        else:
            warnings.warn(message)


@pytest.fixture(name="register_posterior_test")
def _register_posterior_test_fixture():
    def _verify_and_register_posterior_test(posterior, expected_posterior_class):
        assert isinstance(posterior, expected_posterior_class)
        TESTED_POSTERIORS.add(expected_posterior_class)

    return _verify_and_register_posterior_test
