# Copyright 2016 James Hensman, Mark van der Wilk,
#                Valentine Svensson, alexggmatthews,
#                PabloLeon, fujiisoup
# Copyright 2017 Artem Artemev @awav
#
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

import tensorflow as tf

from ..core.base import GPflowError
from ..core.base import Build
from ..core.node import Node

from ..core.autoflow import AutoFlow
from ..core.tensor_converter import TensorConverter

from .. import misc
from .. import settings

from .parameter import Parameter
from .dataholders import DataHolder

class Parameterized(Node):

    def __init__(self, name=None):
        super(Parameterized, self).__init__(name=name)
        self._prior_tensor = None

    @property
    def params(self):
        for key, param in self.__dict__.items():
            if not key.startswith('_') and Parameterized._is_param_like(param):
                yield param

    @property
    def non_empty_params(self):
        for param in self.params:
            if isinstance(param, Parameterized) and param.empty:
                continue
            yield param

    @property
    def empty(self):
        parameters = bool(list(self.parameters))
        data_holders = bool(list(self.data_holders))
        return not (parameters or data_holders)

    @property
    def parameters(self):
        for param in self.params:
            if isinstance(param, Parameterized):
                for sub_param in param.parameters:
                    yield sub_param
            elif not isinstance(param, DataHolder):
                yield param

    @property
    def data_holders(self):
        for param in self.params:
            if isinstance(param, Parameterized):
                for sub_param in param.data_holders:
                    yield sub_param
            elif isinstance(param, DataHolder):
                yield param

    @property
    def trainable_parameters(self):
        for parameter in self.parameters:
            if parameter.trainable:
                yield parameter

    @property
    def trainable_tensors(self):
        for parameter in self.trainable_parameters:
            yield parameter.parameter_tensor

    @property
    def prior_tensor(self):
        return self._prior_tensor

    @property
    def feeds(self):
        total_feeds = {}
        for data_holder in self.data_holders:
            holder_feeds = data_holder.feeds
            if holder_feeds is not None:
                total_feeds.update(holder_feeds)
        if not total_feeds:
            return None
        return total_feeds

    @property
    def initializables(self):
        def get_initializables(param_gen, inits):
            for param in param_gen:
                tensors = param.initializables
                if tensors is not None:
                    inits += tensors
        inits = []
        get_initializables(self.parameters, inits)
        get_initializables(self.data_holders, inits)
        return inits

    @property
    def initializable_feeds(self):
        def get_initializable_feeds(param_gen, feeds):
            for param in param_gen:
                param_feeds = param.initializable_feeds
                if param_feeds is not None:
                    feeds.update(param_feeds)
        feeds = {}
        get_initializable_feeds(self.parameters, feeds)
        get_initializable_feeds(self.data_holders, feeds)
        if not feeds:
            return None
        return feeds

    @property
    def graph(self):
        for param in self.params:
            if param.graph is not None:
                return param.graph
        return None

    @property
    def trainable(self):
        for parameter in self.parameters:
            if parameter.trainable:
                return True
        return False

    @trainable.setter
    def trainable(self, value):
        for param in self.params:
            param.trainable = value

    def __str__(self):
        return '\n\n'.join([p.__str__() for p in self.parameters])

    def assign(self, values, session=None):
        session = self.enquire_session(session=session, allow_none=True)
        params = {param.full_name: param for param in self.parameters}
        val_keys = set(values.keys())
        if not val_keys.issubset(params.keys()):
            keys_not_found = val_keys.difference(params.keys())
            raise ValueError('Input values are not coherent with parameters. '
                             'There keys are not found: {}.'.format(keys_not_found))
        prev_values = {}
        for key in val_keys:
            try:
                prev_values[key] = params[key].read_value()
                params[key].assign(values[key], session=session)
            except (GPflowError, ValueError) as error:
                for rkey, rvalue in prev_values.items():
                    params[rkey].assign(rvalue, session=session)
                raise error

    def anchor(self):
        for parameter in self.trainable_parameters:
            parameter.assign(parameter.read_value())

    def read_trainables(self, session=None):
        session = self.enquire_session(session, allow_none=True)
        return [param.read_value(session) for param in self.trainable_parameters]

    def is_built(self, graph):
        if graph is None:
            raise ValueError('Graph is not specified.')
        statuses = set([param.is_built(graph) for param in self.non_empty_params])
        if Build.NOT_COMPATIBLE_GRAPH in statuses:
            return Build.NOT_COMPATIBLE_GRAPH
        elif Build.NO in statuses:
            return Build.NO
        elif self.prior_tensor is None and list(self.parameters):
            return Build.NO
        return Build.YES

    def set_trainable(self, value, graph=None):
        for param in self.params:
            param.set_trainable(value, graph=graph)

    def initialize(self, session=None):
        session = self.enquire_session(session)
        initializables = self.initializables
        if initializables:
            init = tf.variables_initializer(initializables)
            session.run(init, feed_dict=self.initializable_feeds)

    @staticmethod
    def _is_param_like(value):
        return isinstance(value, (Parameter, Parameterized))

    @staticmethod
    def _tensor_mode_parameter(obj):
        if isinstance(obj, Parameter):
            if isinstance(obj, DataHolder):
                return obj.parameter_tensor
            return obj.constrained_tensor

    def _clear(self):
        self._prior_tensor = None
        AutoFlow.clear_autoflow(self)
        self._reset_name()
        for param in self.params:
            param._clear()  # pylint: disable=W0212

    def _build(self):
        for param in self.params:
            param._build_with_name_scope() # pylint: disable=W0212
        priors = []
        for param in self.params:
            if not isinstance(param, DataHolder):
                if isinstance(param, Parameterized) and param.prior_tensor is None:
                    continue
                priors.append(param.prior_tensor)
        self._prior_tensor = self._build_prior(priors)

    def _build_prior(self, prior_tensors):
        """
        Build a tf expression for the prior by summing all child-parameter priors.
        """
        # TODO(@awav): What prior must represent empty list of parameters?
        if not prior_tensors:
            return tf.constant(0, dtype=settings.tf_float)
        return tf.add_n(prior_tensors, name='prior')

    def _set_param(self, name, value):
        object.__setattr__(self, name, value)
        value.set_parent(self)
        value.set_name(name)

    def _get_param(self, name):
        return getattr(self, name)

    def _update_param_attribute(self, name, value):
        param = self._get_param(name)
        if Parameterized._is_param_like(value):
            if param is value:
                return
            if self.is_built_coherence(value.graph) is Build.YES:
                raise GPflowError('Parameterized object is built.')
            self._set_param(name, value)
            param.set_parent()
            param.set_name()
        elif isinstance(param, Parameter) and misc.is_valid_param_value(value):
            param.assign(value, session=self.session)
        else:
            msg = '"{0}" type cannot be assigned to "{1}".'
            raise ValueError(msg.format(type(value), name))

    def __getattribute__(self, name):
        attr = misc.get_attribute(self, name)
        if TensorConverter.tensor_mode(self) and isinstance(attr, Parameter):
            return Parameterized._tensor_mode_parameter(attr)
        return attr

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return

        if self.root is value:
            raise ValueError('Cannot be assigned as parameter to itself.')

        if key in self.__dict__.keys():
            if Parameterized._is_param_like(getattr(self, key)):
                self._update_param_attribute(key, value)
                return

        if Parameterized._is_param_like(value):
            if not self.empty and self.is_built_coherence(value.graph) is Build.YES:
                raise GPflowError('Cannot be added to assembled node.')
            value.set_parent(self)
            value.set_name(key)

        object.__setattr__(self, key, value)
