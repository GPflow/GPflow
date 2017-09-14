# Copyright 2016 James Hensman, Mark van der Wilk,
#                Valentine Svensson, alexggmatthews,
#                PabloLeon, fujiisoup
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
import functools
import numpy as np
import tensorflow as tf

from .base import IPrior, ITransform
from .base import Build, CompilableNode
from .transforms import Identity

from .misc import GPflowError
from .misc import is_number, is_tensor, is_valid_param_value
from .misc import add_to_trainables, remove_from_trainables
from .misc import normalize_dtype, get_tensor_by_name

from .misc import FLOAT_TYPE, NP_FLOAT_TYPE


_TENSOR_MODE_ATTRIBUTE = '_tensor_mode'


class Param(CompilableNode):
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

    def __init__(self, value, name=None, transform=Identity(), prior=None, trainable=False):
        value = self._valid_param_value(value)
        super(Param, self).__init__(name)
        self._param_tensor = None
        self._prior_tensor = None
        self._externally_defined = False

        self.prior = prior
        self.trainable = trainable
        self.transform = transform

        if is_tensor(value):
            self._externally_defined = True
            self._param_tensor = value
        else:
            self._initial_value = value.copy()

    @property
    def is_tensor_mode(self):
        if self.parent is not self:
            return self.parent.is_tensor_mode
        return False

    @property
    def shape(self):
        if self.param_tensor is not None:
            return self.param_tensor.shape
        return self._initial_value.shape

    @property
    def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return self._initial_value.size

    @property
    def param_tensor(self):
        return self._param_tensor

    @property
    def prior_tensor(self):
        return self._prior_tensor

    @property
    def graph(self):
        if self.param_tensor is None:
            return None
        return self.param_tensor.graph

    def is_built(self, graph=None):
        if graph is None:
            raise ValueError('Graph is not specified.')
        if self.graph and self.graph is not graph:
            return Build.NOT_COMPATIBLE_GRAPH
        elif self.prior_tensor is None:
            return Build.NO
        return Build.YES

    def initialize(self, session=None):
        session = self.enquire_session(session)
        if isinstance(self.param_tensor, tf.Variable):
            init = tf.variables_initializer([self.param_tensor])
            session.run(init)

    def clear(self):
        super(Param, self).clear()
        self._param_tensor = None
        self._prior_tensor = None
        self._externally_defined = False

    def set_trainable(self, value, graph=None):
        if not isinstance(value, bool):
            raise TypeError('Fixed property value must be boolean.')

        if self._externally_defined:
            raise GPflowError('Externally defined parameter tensor is not modifiable.')

        graph = self.enquire_graph(graph)
        is_built = self.is_built_coherence(graph)

        if is_built is Build.YES and self.trainable == value:
            return

        object.__setattr__(self, 'trainable', value)

        if is_built is Build.YES:
            if value:
                remove_from_trainables(self.param_tensor, graph)
            else:
                add_to_trainables(self.param_tensor, graph)

    def assign(self, value, session=None):
        if self._externally_defined:
            raise GPflowError("Externally defined parameter tensor is not modifiable.")
        value = self._valid_param_value(value)
        if self.shape != value.shape:
            raise GPflowError('Assigning value has different shape.')
        session = self.enquire_session(session, allow_none=True)
        self._initial_value[...] = value
        if session and self.is_built_coherence(session.graph) is Build.YES:
            self.param_tensor.load(self._initial_value, session=session)

    def read_value(self, session=None):
        session = self.enquire_session(session, allow_none=True)
        if session:
            is_built = self.is_built_coherence(session.graph)
            if is_built is Build.YES:
                return session.run(self.param_tensor)
        return self._initial_value

    @staticmethod
    def _valid_param_value(value):
        if not is_valid_param_value(value):
            raise ValueError('The value must be either a tensorflow '
                             'variable, an array or a scalar.')
        if is_number(value):
            value = np.array(value, dtype=NP_FLOAT_TYPE)
        return value

    def _build(self):
        self._param_tensor = self._build_param()
        self._prior_tensor = self._build_prior()

    def _build_param(self):
        if self._externally_defined:
            ## Double check for externally created graph
            #if self.graph is not tf.get_default_graph():
            #    raise GPflowError("Externally defined tensor uses different graph.")
            return self.param_tensor

        name = self.full_name + '/variable'
        tensor = get_tensor_by_name(name, graph=self.graph)
        if tensor is not None:
            return tensor

        init = tf.constant_initializer(self._initial_value, dtype=FLOAT_TYPE)
        return tf.get_variable(name, shape=self.shape, initializer=init, dtype=FLOAT_TYPE)

    def _build_prior(self):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if not is_tensor(self.param_tensor):  # pragma: no cover
            raise GPflowError("Parameter's tensor is not compiled.")

        prior_name = 'prior'

        if self.prior is None:
            return tf.constant(0.0, FLOAT_TYPE, name=prior_name)

        var = self.param_tensor
        log_jacobian = self.transform.tf_log_jacobian(var)
        transformed_var = self.transform.tf_forward(var)
        logp_var = self.prior.logp(transformed_var)
        return tf.add(logp_var, log_jacobian, name=prior_name)

    def _set_parameter_attribute(self, attr, value):
        if attr is self.ParamAttribute.TRAINABLE:
            self.set_trainable(value, graph=self.graph)
            return

        is_built = self.is_built_coherence(self.graph)
        if is_built is Build.YES:
            raise GPflowError('Parameter has already been compiled.')

        key = attr.value
        if value is not None and not isinstance(value, attr.interface):
            msg = 'Attribute "{0}" must implement interface "{1}".'
            raise GPflowError(msg.format(key, attr.interface))
        object.__setattr__(self, key, value)

    def _html_table_rows(self, name_prefix=''):
        html = "<tr>"
        html += "<td>{0}</td>".format(name_prefix + self.name)
        html += "<td>{0}</td>".format(str(self._array).replace('\n', '</br>'))
        html += "<td>{0}</td>".format(str(self.prior))
        html += "<td>{0}</td>".format('[FIXED]' if self.trainable else str(self.transform))
        html += "</tr>"
        return html

    def __setattr__(self, key, value):
        try:
            attr = self.ParamAttribute(key)
            self._set_parameter_attribute(attr, value)
            return
        except ValueError:
            pass
        object.__setattr__(self, key, value)

    def __str__(self, prepend=''):
        return prepend + \
               '\033[1m' + self.name + '\033[0m' + \
               ' transform:' + str(self.transform) + \
               ' prior:' + str(self.prior) + \
               (' [FIXED]' if self.trainable else '') + \
               '\n' + str(self.value())


class DataHolder(CompilableNode):
    def __init__(self, array, on_shape_change='raise'):
        super(DataHolder, self).__init__()
        dtype = normalize_dtype(array)
        self._array = np.asarray(array, dtype=dtype)
        assert on_shape_change in ['raise', 'pass', 'recompile']
        self.on_shape_change = on_shape_change

    def make_tf_array(self):
        self._tf_array = tf.placeholder(dtype=self._get_type(self._array),
                                        shape=[None] * self._array.ndim,
                                        name=self.name)

    def set_data(self, array):
        """
        Setting a data into self._array before any TensorFlow execution.
        If the shape of the data changes, then either:
         - raise an exception
         - raise the recompilation flag.
         - do nothing
        according to the option in self.on_shape_change.
        """
        if self.shape == array.shape:
            self._array[...] = array  # just accept the new values
        else:
            if self.on_shape_change == 'raise':
                raise ValueError("The shape of this data must not change. \
                                  (perhaps make the model again from scratch?)")
            elif self.on_shape_change == 'recompile':
                self._array = array.copy()
                self.root._needs_recompile = True
            elif self.on_shape_change == 'pass':
                self._array = array.copy()
            else:
                raise ValueError('invalid option')  # pragma: no cover

    @property
    def value(self):
        return self._array.copy()

    @property
    def size(self):
        return self._array.size

    @property
    def shape(self):
        return self._array.shape

    def __str__(self, prepend='Data:'):
        return prepend + \
               '\033[1m' + self.name + '\033[0m' + \
               '\n' + str(self.value)


class Parameterized(CompilableNode):
    def __init__(self, name=None):
        super(Parameterized, self).__init__(name=name)
        self._prior_tensor = None

    @property
    def is_tensor_mode(self):
        return getattr(self, _TENSOR_MODE_ATTRIBUTE, False)

    @property
    def params(self):
        for key, param in self.__dict__.items():
            if not key.startswith('_') and isinstance(param, (Param, Parameterized)):
                yield param

    @property
    def trainable_params(self):
        for param in self.params:
            if param.trainable:
                continue
            if isinstance(param, Parameterized):
                for sub_param in param.trainable_params:
                    if not sub_param.trainable:
                        yield sub_param
            elif isinstance(param, Param):
                yield param

    @property
    def trainable_tensors(self):
        return [param.param_tensor for param in self.trainable_params]

    @property
    def prior_tensor(self):
        return self._prior_tensor

    def is_built(self, graph=None):
        if graph is None:
            raise ValueError('Graph is not specified.')
        param_graphs = set([param.graph for param in self.params])
        if not param_graphs or None in param_graphs and param_graphs.issubset([None, graph]):
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
        for param in self.params:
            if not param.trainable:
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
        variables = [param.param_tensor for param in self.params
                     if isinstance(param.param_tensor, tf.Variable)]
        if variables:
            init = tf.variables_initializer(variables)
            session.run(init)

    def clear(self):
        super(Parameterized, self).clear()
        for param in self.params:
            param.clear()
        self._prior_tensor = None

    # TODO(awav): # pylint: disable=W0511
    #def randomize(self, distributions={}, skiptrainable=True):
    #    """
    #    Calls randomize on all parameters in model hierarchy.
    #    """
    #    for param in self.sorted_params:
    #        param.randomize(distributions, skiptrainable)

    def _build(self):
        for param in self.params:
            param._build_with_name_scope() # pylint: disable=W0212
        self._prior_tensor = self._build_prior()

    def _build_prior(self):
        """
        Build a tf expression for the prior by summing all child-parameter priors.
        """
        return tf.add_n([param.prior_tensor for param in self.params], name='prior')

    def _update_param_attribute(self, key, value):
        attr = getattr(self, key)
        param_like = (Param, Parameterized)
        if isinstance(value, param_like):
            if self.is_built_coherence(value.graph) is Build.YES:
                raise GPflowError('Attribute cannot be changed in assembled node.')
            attr.set_parent()
            attr.set_name()
            value.set_parent(self)
            value.set_name(key)
            object.__setattr__(self, key, value)
        # elif - DataHolder:
        elif isinstance(attr, Param) and is_valid_param_value(value):
            attr.assign(value, session=self.session)
        else:
            msg = '"{0}" type cannot be assigned to "{1}" attribute.'
            raise ValueError(msg.format(type(value), key))

    def _html_table_rows(self, name_prefix=''):
        """
        Get the rows of the html table for this object
        """
        name_prefix += self.name + '.'
        return ''.join([p._html_table_rows(name_prefix)
                        for p in self.sorted_params])

    def _repr_html_(self):
        """
        Build a small html table for display in the jupyter notebook.
        """
        html = ["<table id='params' width=100%>"]

        # build the header
        header = "<tr>"
        header += "<td>Name</td>"
        header += "<td>values</td>"
        header += "<td>prior</td>"
        header += "<td>constraint</td>"
        header += "</tr>"
        html.append(header)

        html.append(self._html_table_rows())

        html.append("</table>")
        return ''.join(html)

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
            if self.is_built_coherence(value.graph) is Build.YES:
                raise GPflowError('Attribute cannot be added to assembled node.')
            value.set_name(key)
            value.set_parent(self)
        object.__setattr__(self, key, value)

# TODO(awav)
    def __str__(self, prepend=''):
        prepend += self.name + '.'
        return '\n'.join([p.__str__(prepend) for p in self.params])

class ParamList(Parameterized):
    """
    A list of parameters.

    This allows us to store parameters in a list whilst making them 'visible'
    to the GPflow machinery. The correct usage is

    >>> my_list = GPflow.param.ParamList([Param1, Param2])

    You can then iterate through the list. For example, to compute the sum:
    >>> my_sum = reduce(tf.add, my_list)

    or the sum of the squares:
    >>> rmse = tf.sqrt(reduce(tf.add, map(tf.square, my_list)))

    You can append things:
    >>> my_list.append(GPflow.kernels.RBF(1))

    but only if the are Parameters (or Parameterized objects). You can set the
    value of Parameters in the list:

    >>> my_list = GPflow.param.ParamList([GPflow.param.Param(2)])
    >>> print my_list
    unnamed.item0 transform:(none) prior:None
    [ 2.]
    >>> my_list[0] = 12
    >>> print my_list
    unnamed.item0 transform:(none) prior:None
    [ 12.]

    But you can't change elements of the list by assignment:
    >>> my_list = GPflow.param.ParamList([GPflow.param.Param(2)])
    >>> new_param = GPflow.param.Param(4)
    >>> my_list[0] = new_param # raises exception

    """

    def __init__(self, list_of_params):
        Parameterized.__init__(self)
        assert isinstance(list_of_params, list)
        for item in list_of_params:
            assert isinstance(item, (Param, Parameterized))
            item._parent = self
        self._list = list_of_params

    @property
    def sorted_params(self):
        return self._list

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

    def append(self, item):
        assert isinstance(item, (Param, Parameterized)), \
            "this object is for containing parameters"
        item._parent = self
        self.sorted_params.append(item)

    def __len__(self):
        return len(self._list)

    def __setitem__(self, key, value):
        """
        It's not possible to assign to things in the list, but it is possible
        to set their values by assignment.
        """
        self.sorted_params[key]._array[...] = value


def params_as_tensors(method):
    @functools.wraps(method)
    def tensor_mode_wrapper(obj, *args, **kwargs):
        if not isinstance(obj, (Parameterized, ParamList)):
            raise GPflowError('Tensor mode works only with parmeterized object.')
        have_attr = hasattr(obj, _TENSOR_MODE_ATTRIBUTE)
        prev_value = getattr(obj, _TENSOR_MODE_ATTRIBUTE, False)
        setattr(obj, _TENSOR_MODE_ATTRIBUTE, True)
        try:
            result = method(obj, *args, **kwargs)
        finally:
            if have_attr:
                setattr(obj, _TENSOR_MODE_ATTRIBUTE, prev_value)
            else:
                delattr(obj, _TENSOR_MODE_ATTRIBUTE)
        return result
    return tensor_mode_wrapper
