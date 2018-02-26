# Copyright 2018 Artem Artemev @awav
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


import abc
from datetime import datetime

import h5py
import numpy as np

from .. import misc
from .context import Contexture
from .frames import (DictFrame, FrameFactory, ListFrame, ParameterFrame,
                     ParamListFrame, PrimitiveTypeFrame, PriorFrame,
                     SliceFrame, Struct, TensorFlowFrame, TransformFrame)


class BaseSerializer(Contexture, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def dump(self, pathname, data):
        pass
    
    @abc.abstractmethod
    def load(self, pathname):
        pass


class HDF5Serializer(BaseSerializer):
    _class_field = '__instance_type__'
    _module_field = '__module__'
    _variables_field = '__attributes__'
    _extra_field = '__extra_field__'
    _list_type = 'list'
    _dict_type = 'dict'

    def dump(self, pathname, data_frame):
        with h5py.File(pathname) as h5file:
            meta = h5file.create_group('meta')
            date = datetime.now().isoformat(timespec='seconds')
            version = misc.version()
            meta.create_dataset(name='date', data=date)
            meta.create_dataset(name='version', data=version)
            self._serialize(h5file, 'data', data_frame)
    
    def load(self, pathname):
        with h5py.File(pathname) as h5file:
            version = h5file['meta']['version'].value
            date = h5file['meta']['date'].value
            return self._deserialize(h5file['data'])
    
    def _serialize(self, group, name, data):
        if PrimitiveTypeFrame.support(data) or SliceFrame.support(data):
            kwargs = {}
            if data is None:
                kwargs.update(dict(dtype=h5py.Empty('i')))
            group.create_dataset(name=name, data=data, **kwargs)
        elif ListFrame.support(data):
            list_struct = group.create_group(name)
            list_struct.create_dataset(name=self._class_field, data=self._list_type)
            for i in range(len(data)):
                self._serialize(list_struct, str(i), data[i])
        elif DictFrame.support(data):
            dict_struct = group.create_group(name)
            dict_struct.create_dataset(name=self._class_field, data=self._dict_type)
            for key, value in data.items():
                self._serialize(dict_struct, key, value)
        elif isinstance(data, Struct):
            object_struct = group.create_group(name)
            object_struct.create_dataset(name=self._class_field, data=data.class_name)
            object_struct.create_dataset(name=self._module_field, data=data.module_name)
            self._serialize(object_struct, self._extra_field, data=data.extra)
            self._serialize(object_struct, self._variables_field, data.variables)
        else:
            msg = 'Unknown data type {} passed for serialization at "{}".'
            raise TypeError(msg.format(type(data), name))
    
    def _deserialize(self, item):
        if isinstance(item, h5py.Dataset):
            value = self._h5_value(item)
            if isinstance(value, h5py.Empty):
                return None
            return value
        class_name = self._h5_value(item, key=self._class_field)
        keys = list(item.keys())
        keys.remove(self._class_field)
        result = None
        if class_name == self._list_type:
            result = [None] * len(keys)
            for key in keys:
                index = int(key)
                result[index] = self._deserialize(item[key])
        elif class_name == self._dict_type:
            result = {}
            for key in keys:
                result[key] = self._deserialize(item[key])
        else:
            module_name = self._deserialize(item[self._module_field])
            extra = self._deserialize(item[self._extra_field])
            variables = self._deserialize(item[self._variables_field])
            result = Struct(module_name=module_name,
                            class_name=class_name,
                            variables=variables,
                            extra=extra)
        return result
    
    def _h5_value(self, item, key=None):
        return item.value if key is None else item[key].value
