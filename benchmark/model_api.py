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
Classes and other infrastructure for defining and creating models.

Concrete models factories are found in ``models.py``.
"""
from abc import ABC, abstractmethod
from typing import AbstractSet, Callable

import numpy as np

from benchmark.dataset_api import DatasetReq, XYData
from benchmark.registry import TaggedRegistry
from benchmark.tag import Tag, TagReq
from gpflow.models import GPModel


class ModelTag(Tag["ModelTag"]):
    pass


ModelReq = TagReq[ModelTag]


REGRESSION = ModelTag("REGRESSION")
SPARSE = ModelTag("SPARSE")
VARIATIONAL = ModelTag("VARIATIONAL")


class ModelFactory(ABC):
    """
    A way to instantiate a model.

    The factory itself should be cheap to create, though it may take some time to actually create
    the model.
    """

    name: str
    """
    Name of this model / model factory.
    """

    tags: AbstractSet[ModelTag]
    """
    Tags representing properties / capabilities of this model.
    """

    dataset_req: DatasetReq
    """
    This model is compatible with datasets that has these tags.
    """

    @abstractmethod
    def create_model(self, data: XYData, rng: np.random.Generator) -> GPModel:
        """
        Create the model.

        Any model parameters should be randomly initialised, using the `rng`.
        """


MODEL_FACTORIES: TaggedRegistry[ModelFactory, ModelTag] = TaggedRegistry()


ModelFactoryFn = Callable[[XYData, np.random.Generator], GPModel]
"""
A function that can be used as a :class:`ModelFactory`.

Any model parameters should be randomly initialised, using the ``Generator``.
"""


class FnModelFactory(ModelFactory):
    """
    Adapter from a function to a :class:`ModelFactory`.
    """

    def __init__(
        self, name: str, tags: AbstractSet[ModelTag], dataset_req: DatasetReq, fn: ModelFactoryFn
    ) -> None:
        self.name = name
        self.tags = tags
        self.dataset_req = dataset_req
        self._fn = fn

    def create_model(self, data: XYData, rng: np.random.Generator) -> GPModel:
        assert self.dataset_req.satisfied(data.tags)
        return self._fn(data, rng)


def make_model_factory(
    tags: AbstractSet[ModelTag], dataset_req: DatasetReq
) -> Callable[[ModelFactoryFn], FnModelFactory]:
    """
    Decorator for turning a function into a :class:`ModelFactory`.
    """

    def wrap(fn: ModelFactoryFn) -> FnModelFactory:
        name = fn.__name__
        factory = FnModelFactory(name, tags, dataset_req, fn)
        MODEL_FACTORIES.add(factory)
        return factory

    return wrap
