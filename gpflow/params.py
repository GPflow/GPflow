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

from gpflow import settings

from gpflow.core.base import GPflowError
from gpflow.core.base import Build
from gpflow.core.node import Node
from gpflow.core.base import IPrior, ITransform

from gpflow.core.autoflow import AutoFlow
from gpflow.core.tensor_converter import TensorConverter

from gpflow.misc import is_number, is_tensor
from gpflow.misc import is_valid_param_value, is_tensor_trainable
from gpflow.misc import add_to_trainables, remove_from_trainables
from gpflow.misc import get_variable_by_name, get_attribute

from gpflow.transforms import Identity


class Param(Node):
    class ParamAttribute(enum.Enum):
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

    def __init__(self, value, name=None, transform=None, prior=None, trainable=True):
        value = _valid_input(value)
        super(Param, self).__init__(name)

        self._init_parameter_defaults()
        self._init_parameter_attributes(prior, transform, trainable)
        self._init_parameter_value(value)

    @property
    def shape(self):
        if self.var_tensor is not None:
            return tuple(self.var_tensor.shape.as_list())
        return self._value.shape

    @property
    def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return np.multiply.reduce(self.shape, dtype=np.int32)

    @property
    def var_tensor(self):
        return self._var_tensor

    @property
    def prior_tensor(self):
        return self._prior_tensor

    @property
    def transformed_tensor(self):
        return self._transformed_tensor

    @property
    def graph(self):
        if self.var_tensor is None:
            return None
        return self.var_tensor.graph

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
        if isinstance(self.var_tensor, tf.Variable):
            init = tf.variables_initializer([self.var_tensor])
            session.run(init)

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
                remove_from_trainables(self.var_tensor, graph)
            else:
                add_to_trainables(self.var_tensor, graph)

        object.__setattr__(self, 'trainable', value)

    def assign(self, value, session=None):
        if self._externally_defined:
            raise GPflowError("Externally defined parameter tensor is not modifiable.")
        value = _valid_input(value)
        if self.shape != value.shape:
            raise GPflowError('Assigning value has different shape.')
        session = self.enquire_session(session, allow_none=True)
        self._value[...] = value
        if session and self.is_built_coherence(session.graph) is Build.YES:
            self.var_tensor.load(self._value, session=session)

    def read_value(self, session=None):
        session = self.enquire_session(session, allow_none=True)
        if session:
            is_built = self.is_built_coherence(session.graph)
            if is_built is Build.YES:
                return session.run(self.var_tensor)
        elif self._externally_defined:
            raise GPflowError('Externally defined parameter requires session.')
        return self._value

    def _clear(self):
        self._var_tensor = None           # pylint: disable=W0201
        self._prior_tensor = None         # pylint: disable=W0201
        self._externally_defined = False  # pylint: disable=W0201

    def _build(self):
        self._var_tensor = self._build_parameter()            # pylint: disable=W0201
        self._transformed_tensor = self._build_transformed()  # pylint: disable=W0201
        self._prior_tensor = self._build_prior()              # pylint: disable=W0201

    def _build_parameter(self):
        if self._externally_defined:
            self._check_tensor_trainable(self.var_tensor)
            return self.var_tensor

        name = '/'.join([self.full_name, 'variable'])
        tensor = get_variable_by_name(name, graph=self.graph)
        if tensor is not None:
            self._check_tensor_trainable(tensor)
            return tensor

        init = tf.constant_initializer(self._value, dtype=settings.tf_float)
        return tf.get_variable(
            name,
            shape=self.shape,
            initializer=init,
            dtype=settings.tf_float,
            trainable=self.trainable)

    def _build_transformed(self):
        if not is_tensor(self.var_tensor):  # pragma: no cover
            raise GPflowError("Parameter's tensor is not compiled.")
        return self.transform.tf_forward(self.var_tensor)

    def _build_prior(self):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if not is_tensor(self.transformed_tensor):  # pragma: no cover
            raise GPflowError("Parameter's tensor is not compiled.")

        prior_name = 'prior'

        if self.prior is None:
            return tf.constant(0.0, settings.tf_float, name=prior_name)

        var = self.var_tensor
        log_jacobian = self.transform.tf_log_jacobian(var)
        logp_var = self.prior.logp(self.transformed_tensor)
        return tf.add(logp_var, log_jacobian, name=prior_name)

    def _check_tensor_trainable(self, tensor):
        is_trainable = is_tensor_trainable(tensor)
        if is_trainable != self.trainable:
            tensor_status = 'trainable' if is_trainable else 'not trainable'
            param_status = 'trainable' if self.trainable else 'not'
            msg = 'Externally defined tensor is {0} whilst parameter is {1}.'
            raise GPflowError(msg.format(tensor_status, param_status))

    def _init_parameter_defaults(self):
        self._var_tensor = None
        self._prior_tensor = None
        self._transformed_tensor = None
        self._externally_defined = False

    def _init_parameter_value(self, value):
        if is_tensor(value):
            is_trainable = is_tensor_trainable(value)
            if is_trainable != self.trainable:
                status = 'trainable' if is_trainable else 'not trainable'
                ValueError('Externally defined tensor is {0}.'.format(status))
            self._externally_defined = True
            self._var_tensor = value
        else:
            self._value = value.copy()

    def _init_parameter_attributes(self, prior, transform, trainable):
        if transform is None:
            transform = Identity()
        self.prior = prior          # pylint: disable=W0201
        self.transform = transform  # pylint: disable=W0201
        self.trainable = trainable  # pylint: disable=W0201

    def _set_parameter_attribute(self, attr, value):
        if attr is self.ParamAttribute.TRAINABLE:
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
            attr = self.ParamAttribute(name)
            self._set_parameter_attribute(attr, value)
            return
        except ValueError:
            pass
        object.__setattr__(self, name, value)

    def __str__(self, prepend=''):
        return prepend + \
               '\033[1m' + self.name + '\033[0m' + \
               ' transform:' + str(self.transform) + \
               ' prior:' + str(self.prior) + \
               (' [FIXED]' if self.trainable else '') + \
               '\n' + str(self.read_value())


class DataHolder(Param):
    def __init__(self, value, name=None):
        super(DataHolder, self).__init__(value, name=name)

    @property
    def trainable(self):
        return False

    def set_trainable(self, _value, graph=None):
        raise NotImplementedError('Data holder cannot be fixed.')

    def is_built(self, graph):
        if graph is None:
            raise ValueError('Graph is not specified.')
        if self.graph is not None:
            if self.graph is not graph:
                return Build.NOT_COMPATIBLE_GRAPH
            return Build.YES
        return Build.NO

    def _clear(self):
        self._var_tensor = None  # pylint: disable=W0201

    def _build(self):
        self._var_tensor = self._build_parameter()  # pylint: disable=W0201

    def _init_parameter_defaults(self):
        self._var_tensor = None
        self._externally_defined = False

    def _init_parameter_attributes(self, _prior, _transform, _trainable):
        pass

    def _set_parameter_attribute(self, attr, value):
        raise NotImplementedError('Data holder does not have parameter attributes.')

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class Parameterized(Node):

    def __init__(self, name=None):
        super(Parameterized, self).__init__(name=name)
        self._prior_tensor = None

    @property
    def params(self):
        for key, param in self.__dict__.items():
            if not key.startswith('_') and isinstance(param, (Param, Parameterized)):
                yield param

    @property
    def non_empty_params(self):
        for param in self.params:
            if isinstance(param, Parameterized) and param.empty:
                continue
            yield param

    @property
    def empty(self):
        return len(list(self.params)) == 0

    @property
    def parameters(self):
        for param in self.params:
            if isinstance(param, Param):
                yield param
            elif isinstance(param, Parameterized):
                for sub_param in param.parameters:
                    yield sub_param

    @property
    def data_holders(self):
        for data_holder in self.parameters:
            if isinstance(data_holder, DataHolder):
                yield data_holder

    @property
    def trainable_parameters(self):
        for parameter in self.parameters:
            if parameter.trainable:
                yield parameter

    @property
    def trainable_tensors(self):
        for parameter in self.trainable_parameters:
            yield parameter.var_tensor

    @property
    def prior_tensor(self):
        return self._prior_tensor

    def is_built(self, graph):
        if graph is None:
            raise ValueError('Graph is not specified.')
        param_graphs = set([param.graph for param in self.non_empty_params])
        if not param_graphs:
            return Build.YES
        if None in param_graphs and param_graphs.issubset([None, graph]):
            return Build.NO
        elif graph not in param_graphs:
            return Build.NOT_COMPATIBLE_GRAPH
        elif self.prior_tensor is None:
            return Build.NO
        return Build.YES

    @property
    def graph(self):
        for param in self.params:
            if param.graph is not None:
                return param.graph
        return None

    @property
    def trainable(self):
        for parameter in self.parameters:
            if not parameter.trainable:
                return False
        return True

    @trainable.setter
    def trainable(self, value):
        for param in self.params:
            param.trainable = value

    def set_trainable(self, value, graph=None):
        for param in self.params:
            param.set_trainable(value, graph=graph)

    def initialize(self, session=None):
        session = self.enquire_session(session)
        variables = [parameter.var_tensor for parameter in self.parameters
                     if isinstance(parameter.var_tensor, tf.Variable)]
        if variables:
            init = tf.variables_initializer(variables)
            session.run(init)

    # TODO(awav): # pylint: disable=W0511
    #def randomize(self, distributions={}, skiptrainable=True):
    #    """
    #    Calls randomize on all parameters in model hierarchy.
    #    """
    #    for param in self.sorted_params:
    #        param.randomize(distributions, skiptrainable)

    def _clear(self):
        self._prior_tensor = None
        AutoFlow.clear_autoflow(self)
        for param in self.params:
            param._clear()  # pylint: disable=W0212

    def _build(self):
        for param in self.params:
            param._build_with_name_scope() # pylint: disable=W0212
        self._prior_tensor = self._build_prior()

    def _build_prior(self):
        """
        Build a tf expression for the prior by summing all child-parameter priors.
        """
        priors = []
        for param in self.params:
            if isinstance(param, Param) and not isinstance(param, DataHolder):
                priors.append(param.prior_tensor)
            elif isinstance(param, Parameterized) and not param.empty:
                priors.append(param.prior_tensor)
        return tf.add_n(priors, name='prior')

    def _update_param_attribute(self, name, value):
        attr = getattr(self, name)
        param_like = (Param, Parameterized)
        if isinstance(value, param_like):
            if self.is_built_coherence(value.graph) is Build.YES:
                raise GPflowError('Built node cannot be changed.')
            attr.set_parent()
            attr.set_name()
            value.set_parent(self)
            value.set_name(name)
            object.__setattr__(self, name, value)
        elif isinstance(attr, Param) and is_valid_param_value(value):
            attr.assign(value, session=self.session)
        else:
            msg = '"{0}" type cannot be assigned to "{1}" attribute.'
            raise ValueError(msg.format(type(value), name))

    def __getattribute__(self, name):
        if TensorConverter.tensor_mode(self):
            attr = get_attribute(self, name)
            if isinstance(attr, Param):
                if isinstance(attr, DataHolder):
                    return attr.var_tensor
                return attr.transformed_tensor
        return get_attribute(self, name)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return

        param_like = (Param, Parameterized)
        if key in self.__dict__.keys():
            if isinstance(getattr(self, key), param_like):
                self._update_param_attribute(key, value)
                return
        if isinstance(value, param_like):
            if not self.empty and self.is_built_coherence(value.graph) is Build.YES:
                raise GPflowError('Attribute cannot be added to assembled node.')
            value.set_name(key)
            value.set_parent(self)
        object.__setattr__(self, key, value)


class ParamList(Parameterized):
    def __init__(self, list_of_params, name=None):
        super(ParamList, self).__init__(name=None)
        if not isinstance(list_of_params, list):
            raise ValueError('Not acceptable argument type for list_of_params.')
        for i, item in enumerate(list_of_params):
            if not isinstance(item, (Param, Parameterized)):
                raise ValueError('Not acceptable item type: {0}.'.format(type(item)))
            item.set_parent(self)
            item.set_name('{index}/{name}'.format(index=i, name=item.name))
        self._list = list_of_params

    @property
    def params(self):
        for item in self._list:
            yield item

    def append(self, item):
        if not isinstance(item, (Param, Parameterized)):
            raise ValueError('Not acceptable item type: {0}.'.format(type(item)))
        length = self.__len__()
        item.set_parent(self)
        item.set_name('{index}/{name}'.format(index=length, name=item.name))
        self._list.append(item)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        """
        If tf mode is off, this simply returns the corresponding Param .

        If tf mode is on, all items will appear as their tf
        representations.
        """
        o = self.sorted_params[key]
        if isinstance(o, Param) and self._tf_mode:
            return o._tf_array
        return o

    def __setitem__(self, key, value):
        """
        It's not possible to assign to things in the list, but it is possible
        to set their values by assignment.
        """
        self.params[key]._array[...] = value


def _valid_input(value):
    if not is_valid_param_value(value):
        raise ValueError('The value must be either a tensorflow '
                         'variable, an array or a scalar.')
    if is_number(value):
        value = np.array(value, dtype=settings.np_float)
    return value
