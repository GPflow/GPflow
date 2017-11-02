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

# pylint: disable=no-self-use
# pylint: disable=too-few-public-methods

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from ..core.base import Build
from ..core.base import GPflowError


class Optimizer:
    @abc.abstractmethod
    def minimize(self, model, **kwargs):
        raise NotImplementedError()

    def _pop_session(self, model, kwargs):
        session = kwargs.pop('session', model.session)
        if session is None:
            raise ValueError('Session is not specified.')
        if model.is_built_coherence(session.graph) is Build.NO:
            raise GPflowError('Session has different graph.')
        return session

    def _pop_var_list(self, model, kwargs):
        return list(set(model.trainable_tensors).union(kwargs.pop('var_list', [])))

    def _pop_feed_dict(self, kwargs):
        return kwargs.pop('feed_dict', {})

    def _pop_maxiter(self, kwargs, default=1000):
        return kwargs.pop('maxiter', default)
