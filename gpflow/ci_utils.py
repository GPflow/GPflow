# Copyright 2017-2019 GPflow
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

# pylint: skip-file

import os


def is_continuous_integration():
    return os.environ.get('CI', None) is not None


def ci_niter(n: int, test_n: int = 2):
    return test_n if is_continuous_integration() else n


def ci_range(n: int, test_n: int = 2):
    return range(ci_niter(n, test_n))


def ci_list(lst: list, test_n=2):
    return lst[:test_n] if is_continuous_integration() else lst
