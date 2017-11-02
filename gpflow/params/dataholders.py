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

from .. import misc

from ..core.base import GPflowError
from ..core.base import Build

from .parameter import Parameter


class DataHolder(Parameter):
    def __init__(self, value, name=None, dtype=None):
        self._dataholder_tensor = None
        super(DataHolder, self).__init__(value=value, name=name, dtype=dtype)

    @property
    def trainable(self):
        return False

    @property
    def parameter_tensor(self):
        return self._dataholder_tensor

    @property
    def shape(self):
        if self.parameter_tensor is not None:
            return tuple(self.parameter_tensor.shape.as_list())
        return self._value.shape

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

    def _parameter_name(self):
        return misc.tensor_name(self.hidden_full_name, 'dataholder')

    def _clear(self):
        self._reset_name()
        self._initial_value_tensor = None
        self._dataholder_tensor = None

    def _build(self):
        self._dataholder_tensor = self._build_parameter()  # pylint: disable=W0201

    def _init_parameter_defaults(self):
        self._initial_value_tensor = None
        self._dataholder_tensor = None

    def _init_parameter_attributes(self, _prior, _transform, _trainable):
        pass

    def _set_parameter_attribute(self, attr, value):
        raise NotImplementedError('Data holder does not have parameter attributes.')

    def _read_parameter_tensor(self, session):
        return session.run(self._dataholder_tensor)

    def _apply_transform(self, value):
        return value

    def _set_parameter_tensor(self, tensor):
        self._dataholder_tensor = tensor

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __str__(self):
        return self._format_parameter(shape=self.shape)


class FormlessData(DataHolder):
    def __init__(self, value, name=None):
        if not misc.is_valid_param_value(value) or misc.is_tensor(value):
            raise ValueError('The value must be either an array or a scalar.')
        super(FormlessData, self).__init__(value, name=name)

    @property
    def feeds(self):
        if self.parameter_tensor is None:
            return {self.parameter_tensor: self._value}
        return None

    @property
    def initializables(self):
        return None

    def initialize(self, session=None):
        pass

    def _build_parameter(self):
        dtype = self._value.dtype
        name = self._parameter_name()
        return tf.placeholder(dtype, shape=None, name=name)

    def _parameter_name(self):
        name = 'formlessdata'
        if self.parent is self:
            return misc.tensor_name(self.hidden_full_name, name)
        return name


class Minibatch(DataHolder):
    def __init__(self, value, batch_size=1, shuffle=True,
                 seed=None, name=None, dtype=None):
        if not misc.is_valid_param_value(value) or misc.is_tensor(value):
            raise ValueError('The value must be either an array or a scalar.')

        super(Minibatch, self).__init__(value, name=name, dtype=dtype)

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed

    @property
    def initializables(self):
        return [self._iterator_tensor]

    @property
    def initializable_feeds(self):
        if self._dataholder_tensor is None:
            return None
        return {self._cache_tensor: self._value,
                self._batch_size_tensor: self._batch_size}

    def set_batch_size(self, size, session=None):
        self._batch_size = size
        session = self.enquire_session(session, allow_none=True)
        if session is not None:
            self.initialize(session=session)

    def _clear(self):
        self._reset_name()
        self._cache_tensor = None
        self._batch_size_tensor = None
        self._dataholder_tensor = None
        self._iterator_tensor = None
        self._shuffle = True
        self._batch_size = 1
        self._seed = None

    def _build(self):
        self._cache_tensor = self._build_placeholder_cache()
        self._dataholder_tensor = self._build_dataholder()

    def _build_placeholder_cache(self):
        value = self._value
        return tf.placeholder(dtype=value.dtype, shape=value.shape, name='minibatch_init')

    def _build_dataholder(self):
        if self._cache_tensor is None:
            raise GPflowError("Minibatch state corrupted.")
        from tensorflow.contrib.data import Dataset
        data = Dataset.from_tensor_slices(self._cache_tensor)
        data = data.repeat()
        if self._shuffle:
            shape = self._value.shape
            data = data.shuffle(buffer_size=shape[0], seed=self._seed)
        self._batch_size_tensor = tf.placeholder(tf.int64, shape=())
        data = data.batch(batch_size=self._batch_size_tensor)
        self._iterator_tensor = data.make_initializable_iterator()
        name = self._parameter_name()
        return self._iterator_tensor.get_next(name=name)

    def _init_parameter_defaults(self):
        self._cache_tensor = None
        self._batch_size_tensor = None
        self._dataholder_tensor = None
        self._iterator_tensor = None
        self._shuffle = True
        self._batch_size = 1
        self._seed = None

    def _parameter_name(self):
        name = 'minibatch'
        if self.parent is self:
            return misc.tensor_name(self.hidden_full_name, name)
        return name
