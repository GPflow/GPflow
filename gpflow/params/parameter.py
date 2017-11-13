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

from ..core.errors import GPflowError
from ..core.compilable import Build
from ..core.node import Node
from ..core.base import IPrior, ITransform

from .. import misc

from ..transforms import Identity


class Parameter(Node):
    """
    Paramater class is a cornerstone of the GPflow package. It wraps TensorFlow
    variable and its prior and transformation building operations. In GPflow
    computation graph the parameter is a leaf in the tree.

    *Constrained* and *unconstrained* values of the parameter.

    These definitions arise from optimization topic, constrained and unconstrained
    optimization accordingly.

    _Constrained_ value has acceptable value boundaries and user always works with
    such values. It means that when you pass value to create parameter object, the
    parameter assumes that this is _constrained_ value. When user reads value of
    the parameter, the user gets _constrained_ value.

    For example, variance cannot be negative, hence the constraint is [0, ∞).

    ```
    param = gpflow.Param(1.0)  # Here `1.0` is constrained value.
    value = param.read_value() # `value` is constrained value too.
    ```

    *Unconstrained* value doesn't have any limits. The necessity in this value
    is caused by available set of TensorFlow optimizers. In fact, internal
    parameter tensor is unconstrained and will be trained as unconstrained variable.

    ```
    param = gpflow.Param(1.0)  # `1.0` is constrained value.
    param.parameter_tensor     # unconstrained TensorFlow variable.
    ```

    User can manage contraint type which will be used at a parameter. There is special
    *transform* property and option in constructor for that. The user can pass one of
    the implementations of `ITransform` interface. By default Identity transform is
    applied, so that constrained has no difference from unconstrained, and variable
    is simply considered as unconstrained.

    ```
    param = gpflow.Param(1.0) # Identity transform is applied.
    ```

    In example below, the parameter has exponential transform. It means that input
    value is always must be positive [0, ∞), but internal unconstrained tensor has
    no boundaries (∞, ∞).

    ```
    param = gpflow.Param(1.0, transform=gpflow.transforms.Exp())
    param.read_value()
    # 1.0
    gpflow.get_default_session().run(param.parameter_tensor)
    # -1.0000005000290891e-06
    ```

    *Parameter's shape*.

    The parameter's shape is always fixed by default. It means that user is
    not allowed to modify shape when parameter's tensors are built. User can
    modify default behavior by setting up `fix_shape` option to `False`, the
    parameter will be able to change its shape and internal parameter's tensor
    will have floating shape - None. *NOTE: trainable parameters with floating
    shape cannot be trained by a bunch of TensorFlow optimizers like RMSProp,
    Adam as they require shape information for trainable variables in advance of
    optimizer's tensors construction.*

    :param value: Constrained input value. It can be a float, an integer,
        a float or integer like list, numpy array or TensorFlow variable.
    :param transform: Instance of GPflow.ITransform implementation.
    :param prior: Instance of GPflow.IPrior implementation.
    :param trainable: Boolean flag. It indicates whether variables
        will be added to trainbles TensorFlow set or not.
    :param dtype: Type of new parameter.
    :param fix_shape: Default value is `True` and indicates that shape
        of internal tensor will be the same as the shape of the input
        variable and can not be changed. `False` will turn on floating
        shape mode for the tensor.
    :param name: Name of the parameter.

    :raises: ValueError exception if value is not valid. In cases when
        default graph used for building parameter differs from the graph
        used during construction of priors or transformations, the GPflowError
        exception is raised.
    """

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

    def __init__(self, value, transform=None, prior=None,
                 trainable=True, dtype=None, fix_shape=True,
                 name=None):
        self._externally_defined = False
        self._fixed_shape = fix_shape
        value = self._valid_input(value, dtype=dtype)

        super().__init__(name)
        self._init_parameter_defaults()
        self._init_parameter_attributes(prior, transform, trainable)
        self._init_parameter_value(value)

    @property
    def shape(self):
        return self._value.shape

    @property
    def dtype(self):
        if self.parameter_tensor is None:
            return self._value.dtype
        return np.dtype(self.parameter_tensor.dtype.as_numpy_dtype)

    @property
    def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return np.multiply.reduce(self.shape, dtype=np.int32)

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
        if self._externally_defined:
            return None
        return {self._initial_value_tensor: self._apply_transform(self._value)}

    @property
    def graph(self):
        if self.parameter_tensor is None:
            return None
        return self.parameter_tensor.graph

    @property
    def fixed_shape(self):
        return self._fixed_shape

    def fix_shape(self):
        if self._fixed_shape:
            return
        if self.parameter_tensor is not None:
            self.parameter_tensor.set_shape(self.shape)
        self._fixed_shape = True

    def anchor(self, session):
        if session is None:
            ValueError('Session is required when anchoring.')
        if self.trainable:
            value = self.read_value(session=session)
            self.assign(value, session=session)

    def is_built(self, graph):
        if graph is None:
            raise ValueError('Graph is not specified.')
        if self.graph and self.graph is not graph:
            return Build.NOT_COMPATIBLE_GRAPH
        elif self.prior_tensor is None:
            return Build.NO
        return Build.YES

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

    def assign(self, value, session=None, dtype=None, force=True):
        if self._externally_defined:
            raise GPflowError("Externally defined parameter tensor is not modifiable.")

        value = self._valid_input(value, dtype)
        if self.fixed_shape:
            self._value[...] = value.copy()
        else:
            self._value = value.copy()
        if self.is_built_coherence() is Build.YES:
            session = self.enquire_session(session)
            self.is_built_coherence(graph=session.graph)
            self.initialize(session=session, force=force)

    def read_value(self, session=None):
        if session:
            is_built = self.is_built_coherence(session.graph)
            if is_built is Build.YES:
                return self._read_parameter_tensor(session)
        elif self._externally_defined:
            raise GPflowError('Externally defined parameter requires session.')
        return self._value

    def _valid_input(self, value, dtype=None):
        if not misc.is_valid_param_value(value):
            msg = 'The value must be either a tensorflow variable, an array or a scalar.'
            raise ValueError(msg)
        cast = False if dtype is None else True
        is_built = False
        shape = None
        if hasattr(self, '_value'): # The parameter has not initialized yet.
            is_built = self.is_built_coherence() == Build.YES
            shape = self.shape
            inner_dtype = self.dtype
            if dtype is not None and inner_dtype != dtype:
                msg = 'Overriding parameter\'s type "{0}" with "{1}" is not possible.'
                raise ValueError(msg.format(inner_dtype, dtype))
            elif isinstance(value, np.ndarray) and inner_dtype != value.dtype:
                msg = 'The value has different data type "{0}". Parameter type is "{1}".'
                raise ValueError(msg.format(value.dtype, inner_dtype))
            cast = False
            dtype = self._value.dtype
        if misc.is_number(value):
            num_type = misc.normalize_num_type(np.result_type(value).type)
            dtype = num_type if dtype is None else dtype
            value = np.array(value, dtype=dtype)
        elif misc.is_list(value):
            dtype = settings.np_float if dtype is None else dtype
            value = np.array(value, dtype=dtype)
        elif cast:
            value = value.astype(dtype)
        if shape is not None and self.fixed_shape and is_built and shape != value.shape:
            msg = 'Value has different shape. Parameter shape {0}, value shape {1}.'
            raise ValueError(msg.format(shape, value.shape))
        return value

    def _clear(self):
        self._reset_name()
        self._externally_defined = False   # pylint: disable=W0201
        self._initial_value_tensor = None  # pylint: disable=W0201
        self._unconstrained_tensor = None  # pylint: disable=W0201
        self._constrained_tensor = None    # pylint: disable=W0201
        self._prior_tensor = None          # pylint: disable=W0201

    def _build(self):
        unconstrained = self._build_parameter()
        constrained = self._build_constrained(unconstrained)
        prior = self._build_prior(unconstrained, constrained)
        self._unconstrained_tensor = unconstrained  # pylint: disable=W0201
        self._constrained_tensor = constrained      # pylint: disable=W0201
        self._prior_tensor = prior                  # pylint: disable=W0201

    def _build_parameter(self):
        if self._externally_defined:
            self._check_tensor_trainable(self.parameter_tensor)
            return self.parameter_tensor

        name = self._parameter_name()
        tensor = misc.get_variable_by_name(name)
        if tensor is not None:
            raise GPflowError('Tensor with name "{name}" already exists, {tensor}.'
                              .format(name=name, tensor=tensor))

        value = self._apply_transform(self._value)
        shape = value.shape if self.fixed_shape else None
        init = tf.placeholder(self.dtype, shape=shape, name='initial_unconstrained_value')
        self._initial_value_tensor = init
        if self.fixed_shape:
            return tf.get_variable(name, initializer=init, trainable=self.trainable)
        return tf.get_variable(name, initializer=init,
                               validate_shape=False,
                               trainable=self.trainable)


    def _build_constrained(self, parameter_tensor):
        if not misc.is_tensor(parameter_tensor):  # pragma: no cover
            raise GPflowError("Input must be a tensor.")
        return self.transform.forward_tensor(parameter_tensor)

    def _build_prior(self, unconstrained_tensor, constrained_tensor):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if not misc.is_tensor(unconstrained_tensor):
            raise GPflowError("Unconstrained input must be a tensor.")

        if not misc.is_tensor(constrained_tensor):
            raise GPflowError("Constrained input must be a tensor.")

        prior_name = 'prior'

        if self.prior is None:
            return tf.constant(0.0, settings.tf_float, name=prior_name)

        log_jacobian = self.transform.log_jacobian_tensor(unconstrained_tensor)
        logp_var = self.prior.logp(constrained_tensor)
        return tf.squeeze(tf.add(logp_var, log_jacobian, name=prior_name))

    def _check_tensor_trainable(self, tensor):
        is_trainable = misc.is_tensor_trainable(tensor)
        if is_trainable != self.trainable:
            tensor_status = 'trainable' if is_trainable else 'not trainable'
            param_status = 'trainable' if self.trainable else 'not'
            msg = 'Externally defined tensor is {0} whilst parameter is {1}.'
            raise GPflowError(msg.format(tensor_status, param_status))

    def _init_parameter_defaults(self):
        self._initial_value_tensor = None
        self._unconstrained_tensor = None
        self._prior_tensor = None
        self._constrained_tensor = None

    def _init_parameter_value(self, value):
        if misc.is_tensor(value):
            is_trainable = misc.is_tensor_trainable(value)
            if is_trainable != self.trainable:
                status = 'trainable' if is_trainable else 'not trainable'
                raise ValueError('Externally defined tensor is {0}.'.format(status))
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
        return '/'.join([self.hidden_full_name, 'unconstrained'])

    def _set_parameter_tensor(self, tensor):
        self._unconstrained_tensor = tensor

    def _set_parameter_attribute(self, attr, value):
        if attr is self.ParameterAttribute.TRAINABLE:
            self.set_trainable(value, graph=self.graph)
            return

        is_built = self.is_built_coherence(self.graph)
        if is_built is Build.YES:
            raise GPflowError('Parameter "{}" has already been compiled.'.format(self.full_name))

        name = attr.value
        if value is not None and not isinstance(value, attr.interface):
            msg = 'Attribute "{0}" must implement interface "{1}".'
            raise GPflowError(msg.format(name, attr.interface))
        object.__setattr__(self, name, value)

    def _html_table_rows(self, name_prefix=''):
        html = "<tr>"
        html += "<td>{0}</td>".format(name_prefix + self.full_name)
        html += "<td>{0}</td>".format(str(self.read_value()).replace('\n', '</br>'))
        html += "<td>{0}</td>".format(str(self.prior))
        html += "<td>{0}</td>".format('[trainable]' if self.trainable else str(self.transform))
        html += "</tr>"
        return html

    def _format_parameter(self, **kwargs):
        begin = '<{otype} name:\033[1m{name}\033[0m'.format(
            otype=self.__class__.__name__, name=self.full_name)
        if self._externally_defined:
            begin += ' [external tensor]'
        if self._externally_defined:
            end = ' value: unknown'
        else:
            # hijack numpy print options for a moment
            opt = np.get_printoptions()
            np.set_printoptions(threshold=6)
            value_repr = self.read_value().__repr__()
            np.set_printoptions(**opt)

            #reformat numpy repr to our own:
            value_repr = value_repr.replace('\n', '\n ')\
                    .replace('array', '')\
                    .replace('(', '').replace(')', '')
            end = '>\nvalue: {value}'.format(value=value_repr)
        args = {}
        body = ''
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if not value:
                    continue
                arg_value = '[{}]'.format(key)
                body = ' {{{key}}}'.format(key=key) + body
            else:
                arg_value = '{key}:{value}'.format(key=key, value=value)
                body += ' {{{key}}}'.format(key=key)
            args[key] = arg_value
        return (begin + body + end).format(**args)

    def __setattr__(self, name, value):
        try:
            attr = self.ParameterAttribute(name)
            self._set_parameter_attribute(attr, value)
            return
        except ValueError:
            pass
        object.__setattr__(self, name, value)

    def __str__(self):
        return self._format_parameter(
            trainable=self.trainable,
            shape=self.shape,
            transform=self.transform,
            prior=self.prior)
