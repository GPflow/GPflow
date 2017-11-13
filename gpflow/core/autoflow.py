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


from gpflow.misc import get_attribute
from gpflow import settings
import tensorflow as tf
from tensorflow.python.util import nest


class AutoFlow:
    __autoflow_prefix__ = '_autoflow_'

    @classmethod
    def get_autoflow(cls, obj, name):
        if not isinstance(name, str):
            raise ValueError('Name must be string.')
        prefix = cls.__autoflow_prefix__
        autoflow_name = prefix + name
        store = get_attribute(obj, autoflow_name, allow_fail=True, default={})
        if not store:
            setattr(obj, autoflow_name, store)
        return store

    @classmethod
    def clear_autoflow(cls, obj, name=None):
        if name is not None and not isinstance(name, str):
            raise ValueError('Name must be a string.')
        prefix = cls.__autoflow_prefix__
        if name:
            delattr(obj, prefix + name)
        else:
            keys = [attr for attr in obj.__dict__ if attr.startswith(prefix)]
            for key in keys:
                delattr(obj, key)


class TensorType:
    """
    Represents the type of a tensor, that is, its data type and shape. Its
    purpose is to create `tf.placeholder`s for `AutoFlow`. Can contain
    additional arguments to pass to the creation of the placeholder.

    It is necessary to stop `tf.python.utils.nest.flatten` from destructuring
    the information about placeholder types.
    """
    def __init__(self, dtype, shape=None, dims=None, **kwargs):
        """
        Can specify shape of tensor either by an integer giving the number
        of dimensions, using `dims`, or a list, using `shape`.

        `dtype` can be a TensorFlow type, or the `float` class. If it is
        `float`, then the data type is `gpflow.settings.tf_float`.
        """
        if dtype is float:
            dtype = settings.tf_float

        if shape is None:
            if dims is not None:
                shape = [None]*dims
        else:
            assert dims is None, ("It is redundant to specify both shape and "
                                  "number of dimensions.")
        self._args = (dtype, shape)
        self._kwargs = kwargs

    def placeholder(self):
        return tf.placeholder(*self._args, **self._kwargs)

    @staticmethod
    def make_structure(structure):
        types = nest.flatten(structure)
        phs = [tt.placeholder() for tt in types]
        return nest.pack_sequence_as(structure, phs)
