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

import warnings

from gpflow.core.base import GPflowError
from gpflow.core.base import Build
from gpflow.models.model import Model

from gpflow.training import optimizer
from gpflow.training import external_optimizer


class ScipyOptimizer(optimizer.Optimizer):
    def __init__(self, model, **kwargs):
        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')

        self._model = model
        if model.is_built_coherence() is Build.NO:
            raise GPflowError('Model is not specified.')

        with model.graph.as_default():
            objective = model.objective
            self._optimizer = external_optimizer.ScipyOptimizerInterface(
                objective, **kwargs)

    def minimize(self, *_args, **kwargs):
        session = self._pop_session(self.model, kwargs)
        if self.model.is_built_coherence(session.graph) is Build.NO:
            raise GPflowError('Model is not specified.')

        try:
            self._optimizer.minimize(session=session, **kwargs)
        except KeyboardInterrupt:
            warnings.warn('Optimization interrupted.')

    @property
    def model(self):
        return self._model
