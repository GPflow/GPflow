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

import enum
import numpy as np
import tensorflow as tf

from .. import settings

from ..core.base import GPflowError
from ..core.base import Build
from ..core.node import Node
from ..core.base import IPrior, ITransform

from .. import misc

from ..transforms import Identity


class Parameter(Node):
    class ParameterAttribute(enum.Enum):
        PRIOR = 'prior'
        TRANSFORM = 'transform'
        TRAINABLE = 'trainable'

        @property
        def interface(self):
            if self.value == self.PRIOR.value:
                return IPrior
            elif self.value == self.TRANSFORM.value:
                return ITransform
            return None

    def __init__(self, value=None, transform=None, prior=None, trainable=True, name=None):
        value = Parameter._valid_input(value)
        super(Parameter, self).__init__(name)

        self._init_parameter_defaults()
        self._init_parameter_attributes(prior, transform, trainable)
        self._init_parameter_value(value)

    @property
    def shape(self):
        if self.parameter_tensor is not None:
            return tuple(self.parameter_tensor.shape.as_list())
        return self._value.shape

    @property
    def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return np.multiply.reduce(self.shape, dtype=np.int32)

    @property
    def dtype(self):
        if self.parameter_tensor is None:
            return self._value.dtype
        return np.dtype(self.parameter_tensor.dtype.as_numpy_dtype)

    @property
    def parameter_tensor(self):
        return self._unconstrained_tensor

    @property
    def unconstrained_tensor(self):
        return self._unconstrained_tensor

    @property
    def constrained_tensor(self):
        return self._constrained_tensor

    @property
    def prior_tensor(self):
        return self._prior_tensor

    @property
    def feeds(self):
        return None

    @property
    def initializables(self):
        if self._externally_defined:
            return None
        return [self.parameter_tensor]

    @property
    def initializable_feeds(self):
        return None

    @property
    def graph(self):
        if self.parameter_tensor is None:
            return None
        return self.parameter_tensor.graph

    def is_built(self, graph):
        if graph is None:
            raise ValueError('Graph is not specified.')
        if self.graph and self.graph is not graph:
            return Build.NOT_COMPATIBLE_GRAPH
        elif self.prior_tensor is None:
            return Build.NO
        return Build.YES

    def initialize(self, session=None):
        session = self.enquire_session(session)
        initializables = self.initializables
        if initializables:
            init = tf.variables_initializer(initializables)
            session.run(init, feed_dict=self.initializable_feeds)

    def set_trainable(self, value, graph=None):
        if not isinstance(value, bool):
            raise TypeError('Fixed property value must be boolean.')

        if self._externally_defined:
            raise GPflowError('Externally defined parameter tensor is not modifiable.')

        graph = self.enquire_graph(graph)
        is_built = self.is_built_coherence(graph)

        if is_built is Build.YES:
            if self.trainable == value:
                return
            elif value:
                misc.add_to_trainables(self.parameter_tensor, graph)
            else:
                misc.remove_from_trainables(self.parameter_tensor, graph)

        object.__setattr__(self, 'trainable', value)

    def assign(self, value, session=None):
        if self._externally_defined:
            raise GPflowError("Externally defined parameter tensor is not modifiable.")
        value = Parameter._valid_input(value)
        if self.is_built_coherence() is Build.YES:
            if self.shape != value.shape:
                raise GPflowError('Value has different shape.')
            session = self.enquire_session(session)
            self.is_built_coherence(graph=session.graph)
            transformed_value = self._apply_transform(value)
            self.parameter_tensor.load(transformed_value, session=session)
        else:
            self._value[...] = value

    def read_value(self, session=None):
        session = self.enquire_session(session, allow_none=True)
        if session:
            is_built = self.is_built_coherence(session.graph)
            if is_built is Build.YES:
                return self._read_parameter_tensor(session)
        elif self._externally_defined:
            raise GPflowError('Externally defined parameter requires session.')
        return self._value

    @staticmethod
    def _valid_input(value):
        if not misc.is_valid_param_value(value):
            raise ValueError('The value must be either a tensorflow '
                             'variable, an array or a scalar.')
        if misc.is_number(value) or misc.is_list(value):
            value = np.array(value)
        return value

    def _clear(self):
        self._externally_defined = False   # pylint: disable=W0201
        self._prior_tensor = None          # pylint: disable=W0201
        self._unconstrained_tensor = None  # pylint: disable=W0201
        self._constrained_tensor = None    # pylint: disable=W0201

    def _build(self):
        self._unconstrained_tensor = self._build_parameter()  # pylint: disable=W0201
        self._constrained_tensor = self._build_constrained()  # pylint: disable=W0201
        self._prior_tensor = self._build_prior()              # pylint: disable=W0201

    def _build_parameter(self):
        if self._externally_defined:
            self._check_tensor_trainable(self.parameter_tensor)
            return self.parameter_tensor

        name = self._parameter_name()
        tensor = misc.get_variable_by_name(name)
        if tensor is not None:
            self._check_tensor_trainable(tensor)
            return tensor

        value = self._apply_transform(self._value)
        init = tf.constant_initializer(value, dtype=settings.tf_float)
        return tf.get_variable(
            name,
            shape=value.shape,
            initializer=init,
            dtype=settings.tf_float,
            trainable=self.trainable)

    def _build_constrained(self):
        if not misc.is_tensor(self.parameter_tensor):  # pragma: no cover
            raise GPflowError("Parameter's unconstrained tensor is not compiled.")
        return self.transform.forward_tensor(self.parameter_tensor)

    def _build_prior(self):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if not misc.is_tensor(self.constrained_tensor):  # pragma: no cover
            raise GPflowError("Parameter's tensor is not compiled.")

        prior_name = 'prior'

        if self.prior is None:
            return tf.constant(0.0, settings.tf_float, name=prior_name)

        log_jacobian = self.transform.log_jacobian_tensor(self.unconstrained_tensor)
        logp_var = self.prior.logp(self.constrained_tensor)
        return tf.squeeze(tf.add(logp_var, log_jacobian, name=prior_name))

    def _check_tensor_trainable(self, tensor):
        is_trainable = misc.is_tensor_trainable(tensor)
        if is_trainable != self.trainable:
            tensor_status = 'trainable' if is_trainable else 'not trainable'
            param_status = 'trainable' if self.trainable else 'not'
            msg = 'Externally defined tensor is {0} whilst parameter is {1}.'
            raise GPflowError(msg.format(tensor_status, param_status))

    def _init_parameter_defaults(self):
        self._unconstrained_tensor = None
        self._prior_tensor = None
        self._constrained_tensor = None
        self._externally_defined = False

    def _init_parameter_value(self, value):
        if misc.is_tensor(value):
            is_trainable = misc.is_tensor_trainable(value)
            if is_trainable != self.trainable:
                status = 'trainable' if is_trainable else 'not trainable'
                ValueError('Externally defined tensor is {0}.'.format(status))
            self._externally_defined = True
            self._set_parameter_tensor(value)
        else:
            self._value = value.copy()

    def _init_parameter_attributes(self, prior, transform, trainable):
        if transform is None:
            transform = Identity()
        self.prior = prior          # pylint: disable=W0201
        self.transform = transform  # pylint: disable=W0201
        self.trainable = trainable  # pylint: disable=W0201

    def _read_parameter_tensor(self, session):
        return session.run(self.constrained_tensor)

    def _apply_transform(self, value):
        return self.transform.backward(value)

    def _parameter_name(self):
        return '/'.join([self.full_name, 'unconstrained'])

    def _set_parameter_tensor(self, tensor):
        self._unconstrained_tensor = tensor

    def _set_parameter_attribute(self, attr, value):
        if attr is self.ParameterAttribute.TRAINABLE:
            self.set_trainable(value, graph=self.graph)
            return

        is_built = self.is_built_coherence(self.graph)
        if is_built is Build.YES:
            raise GPflowError('Parameter has already been compiled.')

        name = attr.value
        if value is not None and not isinstance(value, attr.interface):
            msg = 'Attribute "{0}" must implement interface "{1}".'
            raise GPflowError(msg.format(name, attr.interface))
        object.__setattr__(self, name, value)

    def _html_table_rows(self, name_prefix=''):
        html = "<tr>"
        html += "<td>{0}</td>".format(name_prefix + self.name)
        html += "<td>{0}</td>".format(str(self._array).replace('\n', '</br>'))
        html += "<td>{0}</td>".format(str(self.prior))
        html += "<td>{0}</td>".format('[FIXED]' if self.trainable else str(self.transform))
        html += "</tr>"
        return html

    def __setattr__(self, name, value):
        try:
            attr = self.ParameterAttribute(name)
            self._set_parameter_attribute(attr, value)
            return
        except ValueError:
            pass
        object.__setattr__(self, name, value)

    def __str__(self):
        trainable = "[TRAINABLE]" if self.trainable else ""
        external = "[EXTERNAL TENSOR]" if self._externally_defined else ""
        msg = ' '.join(['{name}',
                        'shape:{shape}',
                        'transform:{transform}',
                        'prior:{prior}',
                        trainable,
                        external])
        return msg.format(name=self.name, shape=self.shape,
                          transform=self.transform, prior=self.transform).strip()
