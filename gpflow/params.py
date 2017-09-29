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

from gpflow.misc import is_number, is_tensor, is_list
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

    def __init__(self, value=None, transform=None, prior=None, trainable=True, name=None):
        value = _valid_input(value)
        super(Param, self).__init__(name)

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
        if isinstance(self.parameter_tensor, tf.Variable):
            init = tf.variables_initializer([self.parameter_tensor])
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
                add_to_trainables(self.parameter_tensor, graph)
            else:
                remove_from_trainables(self.parameter_tensor, graph)

        object.__setattr__(self, 'trainable', value)

    def assign(self, value, session=None):
        if self._externally_defined:
            raise GPflowError("Externally defined parameter tensor is not modifiable.")
        value = _valid_input(value)
        if self.is_built_coherence() is Build.YES:
            if self.shape != value.shape:
                raise GPflowError('Value has different shape.')
            session = self.enquire_session(session)
            self.is_built_coherence(graph=session.graph)
            value = self._convert_to_unconstrained(value)
            self.parameter_tensor.load(value, session=session)
        else:
            self._value[...] = value

    def read_value(self, session=None):
        session = self.enquire_session(session, allow_none=True)
        if session:
            is_built = self.is_built_coherence(session.graph)
            if is_built is Build.YES:
                return self._read_constrained_tensor(session)
        elif self._externally_defined:
            raise GPflowError('Externally defined parameter requires session.')
        return self._value

    def _clear(self):
        self._unconstrained_tensor = None  # pylint: disable=W0201
        self._prior_tensor = None          # pylint: disable=W0201
        self._externally_defined = False   # pylint: disable=W0201

    def _build(self):
        self._unconstrained_tensor = self._build_parameter()  # pylint: disable=W0201
        self._constrained_tensor = self._build_constrained()  # pylint: disable=W0201
        self._prior_tensor = self._build_prior()              # pylint: disable=W0201

    def _build_parameter(self):
        if self._externally_defined:
            self._check_tensor_trainable(self.parameter_tensor)
            return self.parameter_tensor

        name = self._parameter_name()
        tensor = get_variable_by_name(name, graph=self.graph)
        if tensor is not None:
            self._check_tensor_trainable(tensor)
            return tensor

        value = self._convert_to_unconstrained(self._value)
        init = tf.constant_initializer(value, dtype=settings.tf_float)
        return tf.get_variable(
            name,
            shape=self.shape,
            initializer=init,
            dtype=settings.tf_float,
            trainable=self.trainable)

    def _build_constrained(self):
        if not is_tensor(self.parameter_tensor):  # pragma: no cover
            raise GPflowError("Parameter's unconstrained tensor is not compiled.")
        return self.transform.forward_tensor(self.parameter_tensor)

    def _build_prior(self):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if not is_tensor(self.constrained_tensor):  # pragma: no cover
            raise GPflowError("Parameter's tensor is not compiled.")

        prior_name = 'prior'

        if self.prior is None:
            return tf.constant(0.0, settings.tf_float, name=prior_name)

        var = self.parameter_tensor
        log_jacobian = self.transform.log_jacobian_tensor(var)
        logp_var = self.prior.logp(self.constrained_tensor)
        return tf.add(logp_var, log_jacobian, name=prior_name)

    def _check_tensor_trainable(self, tensor):
        is_trainable = is_tensor_trainable(tensor)
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
        if is_tensor(value):
            is_trainable = is_tensor_trainable(value)
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

    def _read_constrained_tensor(self, session):
        return session.run(self.constrained_tensor)

    def _convert_to_unconstrained(self, value):
        return self.transform.backward(value)

    def _parameter_name(self):
        return '/'.join([self.full_name, 'unconstrained'])

    def _set_parameter_tensor(self, tensor):
        self._unconstrained_tensor = tensor

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


class DataHolder(Param):
    def __init__(self, value, name=None):
        super(DataHolder, self).__init__(value, name=name)

    @property
    def trainable(self):
        return False

    @property
    def parameter_tensor(self):
        return self._dataholder_tensor

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
        self._dataholder_tensor = None  # pylint: disable=W0201

    def _build(self):
        self._dataholder_tensor = self._build_parameter()  # pylint: disable=W0201

    def _init_parameter_defaults(self):
        self._dataholder_tensor = None
        self._externally_defined = False

    def _init_parameter_attributes(self, _prior, _transform, _trainable):
        pass

    def _set_parameter_attribute(self, attr, value):
        raise NotImplementedError('Data holder does not have parameter attributes.')

    def _read_constrained_tensor(self, session):
        return session.run(self.parameter_tensor)

    def _convert_to_unconstrained(self, value):
        return value

    def _parameter_name(self):
        return '/'.join([self.full_name, 'dataholder'])

    def _set_parameter_tensor(self, tensor):
        self._dataholder_tensor = tensor


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
        for data_holder in self.params:
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
            yield parameter.parameter_tensor

    @property
    def prior_tensor(self):
        return self._prior_tensor

    def read_trainable_values(self, session=None):
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

    def set_trainable(self, value, graph=None):
        for param in self.params:
            param.set_trainable(value, graph=graph)

    def initialize(self, session=None):
        session = self.enquire_session(session)
        variables = [parameter.parameter_tensor for parameter in self.parameters
                     if isinstance(parameter.parameter_tensor, tf.Variable)]
        data_holders = [data_holder.parameter_tensor for data_holder in self.data_holders
                        if isinstance(data_holder.parameter_tensor, tf.Variable)]
        var_list = variables + data_holders
        if var_list:
            init = tf.variables_initializer(var_list)
            session.run(init)

    # TODO(@awav): # pylint: disable=W0511
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

        # TODO(@awav): What prior must represent empty list of parameters?
        if not priors:
            return None
        return tf.add_n(priors, name='prior')

    def _set_param(self, name, value):
        object.__setattr__(self, name, value)
        value.set_parent(self)
        value.set_name(name)

    def _get_param(self, name):
        return getattr(self, name)

    def _update_param_attribute(self, name, value):
        param = self._get_param(name)
        if _is_param_like(value):
            if param is value:
                return
            if self.is_built_coherence(value.graph) is Build.YES:
                raise GPflowError('Parameterized object is built.')
            self._set_param(name, value)
            param.set_parent()
            param.set_name()
        elif isinstance(param, Param) and is_valid_param_value(value):
            param.assign(value, session=self.session)
        else:
            msg = '"{0}" type cannot be assigned to "{1}".'
            raise ValueError(msg.format(type(value), name))

    def __getattribute__(self, name):
        attr = get_attribute(self, name)
        if TensorConverter.tensor_mode(self) and isinstance(attr, Param):
            return _tensor_mode_parameter(attr)
        return attr

    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return

        if self.root is value:
            raise ValueError('Cannot be assigned as parameter to itself.')

        if key in self.__dict__.keys():
            if _is_param_like(getattr(self, key)):
                self._update_param_attribute(key, value)
                return

        if _is_param_like(value):
            if not self.empty and self.is_built_coherence(value.graph) is Build.YES:
                print('{}'.format(self.empty))
                print('{}'.format(list(self.parameters)))
                print('{}'.format(list(self.data_holders)))
                raise GPflowError('Attribute cannot be added to assembled node.')
            value.set_name(key)
            value.set_parent(self)

        object.__setattr__(self, key, value)


class ParamList(Parameterized):
    def __init__(self, list_of_params, trainable=True, name=None):
        super(ParamList, self).__init__(name=None)
        if not isinstance(list_of_params, list):
            raise ValueError('Not acceptable argument type at list_of_params.')
        self._list = [self._valid_list_input(item, trainable) for item in list_of_params]
        for index, item in enumerate(self._list):
            self._set_param(index, item)

    @property
    def params(self):
        for item in self._list:
            yield item

    def append(self, item):
        if not isinstance(item, Param):
            raise ValueError('Non parameter type cannot be appended to the list.')
        length = self.__len__()
        item.set_parent(self)
        item.set_name(self._item_name(length))
        self._list.append(item)

    def _get_param(self, name):
        return self._list[name]

    def _set_param(self, name, value):
        self._list[name] = value
        value.set_parent(self)
        value.set_name(self._item_name(name))

    def _item_name(self, index):
        return '{name}{index}'.format(name='item', index=index)

    def _valid_list_input(self, value, trainable):
        if not _is_param_like(value):
            try:
                return Param(value, trainable=trainable)
            except ValueError:
                raise ValueError(
                    'A list item must be either parameter, '
                    'tensorflow variable, an array or a scalar.')
        return value

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        param = self._get_param(key)
        if TensorConverter.tensor_mode(self):
            return _tensor_mode_parameter(param)
        return param

    def __setitem__(self, index, value):
        if not isinstance(value, Param):
            raise ValueError('Non parameter type cannot be assigned to the list.')
        if not self.empty and self.is_built_coherence(value.graph) is Build.YES:
            raise GPflowError('ParamList is compiled and items are not modifiable.')
        self._update_param_attribute(index, value)


def _is_param_like(value):
    return isinstance(value, (Param, Parameterized))


def _tensor_mode_parameter(obj):
    if isinstance(obj, Param):
        if isinstance(obj, DataHolder):
            return obj.parameter_tensor
        return obj.constrained_tensor


def _valid_input(value):
    if not is_valid_param_value(value):
        raise ValueError('The value must be either a tensorflow '
                         'variable, an array or a scalar.')
    if is_number(value) or is_list(value):
        value = np.array(value)
    return value
