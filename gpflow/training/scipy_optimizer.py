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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gpflow.core.base import GPflowError
from gpflow.core.base import Build
from gpflow.models.model import Model

from . import optimizer
from . import external_optimizer


class ScipyOptimizer(optimizer.Optimizer):
    def __init__(self, **kwargs):
        self._optimizer_kwargs = kwargs
        self._optimizer = None
        self._model = None

    def minimize(self, model, **kwargs):
        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')

        if model.is_built_coherence() is Build.NO:
            raise GPflowError('Model is not built.')

        session = self._pop_session(model, kwargs)
        self._model = model

        var_list = self._pop_var_list(model, kwargs)
        with model.graph.as_default():
            objective = model.objective
            self._optimizer = external_optimizer.ScipyOptimizerInterface(
                objective, var_list=var_list, **self._optimizer_kwargs)

        self._optimizer.minimize(session=session, **kwargs)

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer
