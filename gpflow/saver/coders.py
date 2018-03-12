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
import sys
from collections import namedtuple
from copy import copy
from enum import Enum
from types import FunctionType

import numpy as np
import tensorflow as tf

from ..core import AutoFlow, Node
from ..params import Parameter, Parameterized, ParamList
from ..priors import Prior
from ..transforms import Transform
from .context import BaseContext, Contexture


class StructField:
    type_field = '__type__'
    data = '__data__'
    module = '__module__'
    class_field = '__class__'
    function_field = '__function__'
    extra = '__extra__'

class StructType(Enum):
    OBJECT = 0
    DICT = 1
    LIST = 2
    FUNCTION = 3
    SLICE = 4


class BaseCoder(Contexture, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def support_encoding(cls, item):
        pass

    @classmethod
    @abc.abstractmethod
    def support_decoding(cls, item):
        pass

    @abc.abstractmethod
    def encode(self, item):
        pass

    @abc.abstractmethod
    def decode(self, item):
        pass


class PrimitiveTypeCoder(BaseCoder):
    @classmethod
    def types(cls):
        return (str, int, float, bool,
                np.string_, np.bytes_,
                np.ndarray, np.bool_,
                np.number, type(None))

    @classmethod
    def support_encoding(cls, item):
        return isinstance(item, cls.types())

    @classmethod
    def support_decoding(cls, item):
        if _is_numpy_object(item):
            return False
        if _is_nan(item) or _is_str(item):
            return True
        return isinstance(item, cls.types())

    def encode(self, item):
        if isinstance(item, str):
            return np.string_(item)
        return numpy_none() if item is None else np.array(item)
    
    def decode(self, item):
        if _is_str(item):
            return _convert_to_string(item)
        return None if _is_nan(item) else item


class TensorFlowCoder(BaseCoder):
    @classmethod
    def support_encoding(cls, item):
        supported_types = (tf.Variable, tf.Tensor, tf.Operation, tf.data.Iterator)
        return isinstance(item, supported_types)

    @classmethod
    def support_decoding(cls, _item):
        pass

    def encode(self, item):
        return numpy_none()

    def decode(self, _item):
        return None


class StructCoder(BaseCoder):
    @classmethod
    @abc.abstractmethod
    def encoding_type(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def decoding_type(cls):
        pass
    
    @classmethod
    def struct(cls, type_name, data, data_dtype=None, shape=None):
        data_dtype = data.dtype if data_dtype is None else data_dtype
        shape = data.shape if shape is None and data.shape else shape
        data_dtype = [data_dtype]
        if shape:
            data_dtype = [data_dtype[0], shape]
        dtype = np.dtype([type_pattern(), (StructField.data, *data_dtype)])
        return np.array((type_name, data), dtype=dtype)

    @classmethod
    def support_encoding(cls, item):
        return isinstance(item, cls.encoding_type())
    
    @classmethod
    def support_decoding(cls, item):
        if not _is_numpy_object(item):
            return False
        fields = item.dtype.fields
        type_field = StructField.type_field
        if not fields or type_field not in fields:
            return False
        return item[type_field] == cls.decoding_type()


class ListCoder(StructCoder):
    @classmethod
    def decoding_type(cls):
        return StructType.LIST.value

    @classmethod
    def encoding_type(cls):
        return list

    def encode(self, item):
        factory = CoderFactory(self.context)
        data = [factory.encode(e) for e in item]
        dtypes_set = set([d.dtype for d in data])
        dtypes_len = len(dtypes_set)
        shape = None
        if not dtypes_len:
            data = numpy_none()
        elif dtypes_len == 1:
            data_dtype = dtypes_set.pop()
            data = np.array(data, dtype=data_dtype)
            shape = (len(data),)
        else:
            data_dtype = [('{}'.format(i), d) for i, d in enumerate(data)]
            data_dtype = _list_of_dtypes(data_dtype)
            data = np.array(tuple(data), dtype=data_dtype)
        return self.struct(self.decoding_type(), data, shape=shape)
        
    def decode(self, item):
        data = item[StructField.data]
        if _is_nan(data):
            return []
        if not data.shape:
            keys = sorted(data.dtype.fields.keys())
            data = [data[k] for k in keys]
        factory = CoderFactory(self.context)
        return [factory.decode(d) for d in data]


class DictCoder(StructCoder):
    @classmethod
    def decoding_type(cls):
        return StructType.DICT.value

    @classmethod
    def encoding_type(cls):
        return dict

    def encode(self, item):
        factory = CoderFactory(self.context)
        pre_data = {k : factory.encode(v) for k, v in item.items()}
        if not pre_data:
            data = numpy_none()
        else:
            data_values = [v for _, v in pre_data.items()]
            data_dtype = _list_of_dtypes(pre_data)
            data = np.array(tuple(data_values), dtype=data_dtype)
        return self.struct(self.decoding_type(), data)
    
    def decode(self, item):
        data = item[StructField.data]
        if _is_nan(data):
            return {}
        factory = CoderFactory(self.context)
        return {k : factory.decode(data[k]) for k in data.dtype.fields.keys()}


class SliceCoder(StructCoder):
    @classmethod
    def decoding_type(cls):
        return StructType.SLICE.value

    @classmethod
    def encoding_type(cls):
        return slice

    def encode(self, item):
        def try_encode(e):
            return numpy_none() if e is None else e
        start = try_encode(item.start)
        stop = try_encode(item.stop)
        step = try_encode(item.step)
        data = np.array([start, stop, step])
        return self.struct(self.decoding_type(), data)

    def decode(self, item):
        data = item[StructField.data]
        def try_decode(e):
            return None if _is_nan(e) else int(e)
        return slice(*map(try_decode, data))


class FunctionCoder(StructCoder):
    @classmethod
    def decoding_type(cls):
        return StructType.FUNCTION.value

    @classmethod
    def encoding_type(cls):
        return FunctionType
    
    def encode(self, item):
        factory = CoderFactory(self.context)
        name = factory.encode(item.__name__)
        module = factory.encode(item.__module__)
        dtype = np.dtype([type_pattern(),
                          (StructField.module, module.dtype),
                          (StructField.function_field, name.dtype)])
        return np.array((self.decoding_type(), module, name), dtype=dtype)
    
    def decode(self, item):
        factory = CoderFactory(self.context)
        module = factory.decode(item[StructField.module])
        name = factory.decode(item[StructField.function_field])
        return _build_type(module, name)


class ObjectCoder(StructCoder):
    @classmethod
    def decoding_type(cls):
        return StructType.OBJECT.value

    @classmethod
    def encoding_type(cls):
        return object
    
    @classmethod
    def support_decoding(cls, item):
        if not super().support_decoding(item):
            return False
        factory = CoderFactory(BaseContext())
        module = factory.decode(item[StructField.module])
        name = factory.decode(item[StructField.class_field])
        item_type = _build_type(module, name)
        return issubclass(item_type, cls.encoding_type())
    
    def encode(self, item):
        name = self._take_class_name(item)
        module = self._take_module_name(item)
        values = self._take_values(item)

        factory = CoderFactory(self.context)
        data = factory.encode(values)

        extra = self._take_extras(item)
        extra_data = factory.encode(extra)

        name = factory.encode(item.__class__.__name__)
        module = factory.encode(item.__module__)
        dtype = np.dtype([type_pattern(),
                          (StructField.module, module.dtype),
                          (StructField.class_field, name.dtype),
                          (StructField.data, data.dtype),
                          (StructField.extra, extra_data.dtype)])
        return np.array((StructType.OBJECT.value, module, name, data, extra_data), dtype=dtype)
    
    def decode(self, item):
        variables = self._decode_attributes(item)
        return self._decode_object(item, variables)
    
    def _take_module_name(self, item):
        return item.__class__.__module__

    def _take_class_name(self, item):
        return item.__class__.__name__
    
    def _take_values(self, item):
        return copy(vars(item))

    def _take_extras(self, item):
        pass
    
    def _transform_values(self, _item, values):
        return CoderFactory(self.context).encode(values)
    
    def _transform_extra(self, item, extra):
        return CoderFactory(self.context).encode(extra)
    
    def _decode_attributes(self, item):
        data = item[StructField.data]
        return CoderFactory(self.context).decode(data)
    
    def _decode_object(self, item, attributes):
        factory = CoderFactory(self.context)
        module = factory.decode(item[StructField.module])
        name = factory.decode(item[StructField.class_field])
        item_type = _build_type(module, name)
        instance = object.__new__(item_type)
        instance.__dict__ = attributes
        return instance


class TransformCoder(ObjectCoder):
    @classmethod
    def encoding_type(cls):
        return Transform


class PriorCoder(ObjectCoder):
    @classmethod
    def encoding_type(cls):
        return Prior


class NodeCoder(ObjectCoder):
    @classmethod
    def encoding_type(cls):
        return Node
    
    def _take_values(self, item):
        values = super()._take_values(item)
        values['_parent'] = None
        return values


class ParameterCoder(NodeCoder):
    @classmethod
    def encoding_type(cls):
        return Parameter
    
    def _take_values(self, item):
        session = self.context.session
        cached_value = np.array(item.read_value(session=session))
        values = super()._take_values(item)
        values['_value'] = cached_value
        return values
    
    def _take_extras(self, item):
        index = item.tf_compilation_index()
        if index is not None:
            if item.index == index:
                return True
            _add_index_to_compilations(self.context, index)
        return None
    
    def _decode_object(self, item, attributes):
        instance = super()._decode_object(item, attributes)
        extra = item[StructField.extra]
        extra = CoderFactory(self.context).decode(extra)
        if extra and self.context.autocompile:
            instance.compile(session=self.context.session)
        return instance


class ParameterizedCoder(NodeCoder):
    @classmethod
    def _encoding_type(cls):
        return Parameterized
    
    def _take_values(self, item):
        values = super()._take_values(item)
        values = {k: v for k, v in values.items() if not k.startswith(AutoFlow.__autoflow_prefix__)}
        return values

    def _decode_object(self, item, attributes):
        instance = super()._decode_object(item, attributes)
        for attr in attributes.values():
            if isinstance(attr, Node):
                setattr(attr, '_parent', instance)
        extra = item[StructField.extra]
        extra = CoderFactory(self.context).decode(extra)
        if extra and self.context.autocompile:
            instance.compile(session=self.context.session)
        return instance

    def _take_extras(self, item):
        if _check_index_in_compilations(self.context, item.index):
            return True
        return None


class ParamListCoder(ParameterizedCoder):
    @classmethod
    def encoding_type(cls):
        return ParamList


class CoderFactory(BaseCoder):
    @classmethod
    def support_decoding(cls, item):
        pass

    @classmethod
    def support_encoding(cls, item):
        pass

    @property
    def coders(self):
        return (PrimitiveTypeCoder,
                TensorFlowCoder,
                FunctionCoder,
                ListCoder,
                DictCoder,
                SliceCoder,
                ParameterCoder,
                ParamListCoder,
                ParameterizedCoder,
                TransformCoder,
                PriorCoder)
    
    def _execute_coder(self, item, coding):
        coders = self.context.coders + self.coders
        for coder in coders:
            if coding == 'encode' and coder.support_encoding(item):
                return coder(self.context).encode(item)
            elif coding == 'decode' and coder.support_decoding(item):
                return coder(self.context).decode(item)
        msg = 'Item "{}" has type {} which does not match any coder at saver for {}.'
        raise TypeError(msg.format(item, type(item), coding))

    def encode(self, item):
        return self._execute_coder(item, 'encode')
    
    def decode(self, item):
        return self._execute_coder(item, 'decode')


# ====================
# Auxillary functions.
# ====================


def type_pattern():
    return (StructField.type_field, np.uint8)


def empty_array():
    return np.array([], np.uint8)


def numpy_none():
    return np.array(np.nan)


def _add_index_to_compilations(context, index):
    compilations = 'compilations'
    if compilations not in context.shared_data:
        context.shared_data[compilations] = set([])
    context.shared_data[compilations].add(index)
    

def _check_index_in_compilations(context, index):
    compilations = 'compilations'
    if compilations not in context.shared_data:
        return False
    return index in context.shared_data[compilations]


def _build_type(module_name, object_name):
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        msg = 'Saver can not find module {}.'
        raise ImportError(msg.format(module_name))
    module = sys.modules[module_name]
    try:
        return module.__dict__[object_name]
    except KeyError:
        msg = 'Saver can not find type {} at module {}.'
        raise KeyError(msg.format(object_name, module_name))


def _list_of_dtypes(values):
    dtypes = []
    if isinstance(values, dict):
        values = values.items()
    for k, v in values:
        if isinstance(v, np.ndarray) and not _is_shapeless(v):
            dtypes.append((k, v.dtype, v.shape))
        else:
            dtypes.append((k, v.dtype))
    return dtypes


def _is_str(value):
    if isinstance(value, (str, np.bytes_)):
        return True
    if hasattr(value, 'dtype'):
        if np.issubdtype(value.dtype, np.bytes_) or np.issubdtype(value.dtype, np.string_):
            return True
    return False


def _is_nan(value):
    if isinstance(value, np.ndarray):
        if value.shape:
            return False
        if np.issubdtype(value.dtype.type, np.floating):
            return np.isnan(value)
    if not isinstance(value, (np.number, float)):
        return False
    return np.isnan(value)


def _is_shapeless(value):
    shape = value.shape
    if not shape or (len(shape) == 1 and shape[0] == 0):
        return True
    return False


def _is_numpy_object(value):
    if not isinstance(value, (np.ndarray, np.void)):
        return False
    return value.dtype.type is np.void


def _convert_to_string(value):
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = np.string_(value)
    return value.decode('utf-8')

