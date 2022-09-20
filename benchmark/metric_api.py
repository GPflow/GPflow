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
Classes and other infrastructure for defining metrics.

Concrete metrics are defined in `metrics.py`.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from benchmark.registry import Named, Registry


class MetricOrientation(Enum):
    """
    Whether we like small or large values.
    """

    LOWER_IS_BETTER = "lower_is_better"
    GREATER_IS_BETTER = "greater_is_better"


@dataclass(frozen=True)
class Metric(Named):
    """
    Metadata about a metric.

    Super useful when plotting it.
    """

    name: str
    pretty_name: str
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    orientation: MetricOrientation
    unit: Optional[str]

    def __repr__(self) -> str:
        return self.name


METRICS: Registry[Metric] = Registry()


def make_metric(**kwargs: Any) -> Metric:
    """
    Create a new :class:`Metric` and register it.

    :param kwargs: Passed on to ``Metric.__init__``.
    """
    result = Metric(**kwargs)
    METRICS.add(result)
    return result
