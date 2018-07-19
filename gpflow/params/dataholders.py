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


import tensorflow as tf

from .. import misc
from ..core.errors import GPflowError
from ..core.compilable import Build
from .parameter import Parameter


class DataHolder(Parameter):
    """
    DataHolder is similar to the Parameter with only difference that
    default values for `fix_shape` and `trainable` options are opposite
    to the Parameter's and it does not have prior and transform options.

    By default shape of data holders is in floating mode and data holder
    does not provide a trainable option at all.

    :param value: Data input value. It can be a float, an integer,
        a float or integer like list, numpy array or TensorFlow variable.
    :param dtype: Type of new data holder.
    :param fix_shape: Default value is `False` and indicates that shape
        of internal tensor does not have specific shape, in other words,
        it is None.
    :param name: Name of the parameter.

    :raises: ValueError exception if value is not valid.
    """

    def __init__(self, value, dtype=None, fix_shape=False, name=None):
        self._dataholder_tensor = None
        super().__init__(value=value, name=name, dtype=dtype, fix_shape=fix_shape)

    @property
    def trainable(self):
        return False

    @property
    def parameter_tensor(self):
        return self._dataholder_tensor

    def set_trainable(self, value, graph=None):
        raise NotImplementedError('Data holder cannot be fixed.')

    def is_built(self, graph):
        if graph is None:
            raise ValueError('Graph is not specified.')
        if self.graph is not None:
            if self.graph is not graph:
                return Build.NOT_COMPATIBLE_GRAPH
            return Build.YES
        return Build.NO

    def as_pandas_table(self):
        column_names = ['class', 'shape', 'fixed_shape', 'value']
        column_values = [self.__class__.__name__, self.shape, self.fixed_shape, self.value]
        column_values = [[value] for value in column_values]
        df = misc.pretty_pandas_table([self.pathname], column_names, column_values)
        return df

    def _parameter_name(self):
        return misc.tensor_name(self.tf_pathname, 'dataholder')

    def _clear(self):
        self.reset_name()
        self._initial_value_tensor = None
        self._dataholder_tensor = None
        self._is_initialized_tensor = None

    def _build(self):
        tensor = self._build_parameter()
        self._dataholder_tensor = tensor
        self._is_initialized_tensor = tf.is_variable_initialized(tensor)

    def _init_parameter_defaults(self):
        self._initial_value_tensor = None
        self._dataholder_tensor = None
        self._is_initialized_tensor = None

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


class Minibatch(DataHolder):
    """
    Minibatch is a special case of data holders. As the name implies the minibatch
    object provides shuffling and endless batching mechanism for input data.
    Minibatch formes batches along zero axe of the input array.

    Minibatch is shape agnostic at zero axe. Once you created a minibatch you can
    vary size of the dataset, but feature shapes must be fixed.

    CAVEAT: Minibatch is not auto-initializable. It means that whenever you switch
    to another session, autoflow methods and optimizers will not be able to
    intialize TensorFlow dataset iterator. You have to call `intialize` method
    for Minibatch explicitly. Simple cases are not affected though.

    ```
    with tf.Session() as session1:
        mini = gpflow.Minibatch(data)

    with tf.Session() as session2:
        # mini.read_value(session=session2) # <<< fails.
        mini.initialize(session=session2)
        mini.read_value(session=session2) # <<< works fine.
    ```

    :param value: Numpy array.
    :param batch_size: Size of the batches.
    :param shuffle: If `True` then input data will be shuffled before batching.
    :param seed: Seed value for TensorFlow random generator.
    :param dtype: Type of new minibatch.
    :param name: Minibatch name.

    :raises: ValueError exception if input value is not a numpy array or a list.
    """

    def __init__(self, value, batch_size=1, shuffle=True,
                 seed=None, dtype=None, name=None):
        if not misc.is_valid_param_value(value) or misc.is_tensor(value):
            raise ValueError('The value must be either an array or a scalar.')

        super().__init__(value, name=name, dtype=dtype)

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        return self.set_batch_size(value)

    @property
    def initializables(self):
        return [self._iterator_tensor]

    @property
    def initializable_feeds(self):
        if self._dataholder_tensor is None:
            return None
        return {self._cache_tensor: self._value,
                self._batch_size_tensor: self._batch_size}

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        if self.graph is not None and self.is_built_coherence():
            raise GPflowError('Minibatch seed cannot be changed when it is built.')
        self._seed = seed

    def set_batch_size(self, size, session=None):
        self._batch_size = size
        session = self.enquire_session(session)
        if session is not None:
            self.initialize(session=session, force=True)

    def _clear(self):
        self.reset_name()
        self._cache_tensor = None
        self._batch_size_tensor = None
        self._dataholder_tensor = None
        self._iterator_tensor = None

    def _build(self):
        initial_tensor = self._build_placeholder_cache()
        self._cache_tensor = initial_tensor
        self._dataholder_tensor = self._build_dataholder(initial_tensor)

    def _build_placeholder_cache(self):
        value = self._value
        return tf.placeholder(dtype=value.dtype, shape=None, name='minibatch_init')

    def _build_dataholder(self, initial_tensor):
        if initial_tensor is None:
            raise GPflowError("Minibatch state corrupted.")
        data = tf.data.Dataset.from_tensor_slices(initial_tensor)
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
            return misc.tensor_name(self.tf_pathname, name)
        return name
