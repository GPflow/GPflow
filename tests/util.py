# Copyright 2017 Artem Artemev @awav
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


# pragma: no cover
# pylint: skip-file


import functools
import contextlib
import tensorflow as tf
import pytest
import os


def is_continuous_integration():
    ci = os.environ.get('CI', '').lower()
    return (ci == 'true') or (ci == '1')


def notebook_niter(n, test_n=2):
    return test_n if is_continuous_integration() else n


def notebook_range(n, test_n=2):
    return range(notebook_niter(n, test_n))


def notebook_list(lst, test_n=2):
    return lst[:test_n] if is_continuous_integration() else lst