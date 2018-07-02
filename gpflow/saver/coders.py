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

"""
Coders module exploits Numpy custom [dtypes](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.rec.html#module-numpy.doc.structured_arrays)
to construct shippable python structures.
Briefly, custom numpy dtypes are structures with any description. Type description includes
a field name, type of the value and optionally value shape. Encoded values as numpy objects
can be serialized saving them as byte arrays or any other suitable approach.

Standard coders can encode/decode:

* Primitive types: int, str, float, bool, NoneType.
* GPflow objects: Parameter, Parameterized, ParamList, Prior, Transform.
* List.
* Dictionary with string keys.
* Numpy array and scalar.
* Function, except lambda and class methods.

"""

import abc
import sys
from collections import namedtuple
from copy import copy
from enum import Enum
from typing import Any, AnyStr, Optional, Dict, List, Union, Type, Tuple
from types import FunctionType

import numpy as np
import tensorflow as tf

from ..core import AutoFlow, Node
from ..params import Parameter, Parameterized, ParamList
from ..priors import Prior
from ..transforms import Transform
from .context import BaseContext, Contexture



class StructField(Enum):
    """Custom np.dtype field names."""
    TYPE = '__type__'
    DATA = '__data__'
    MODULE = '__module__'
    CLASS = '__class__'
    FUNCTION = '__function__'
    EXTRA = '__extra__'



class StructType(Enum):
    """Custom np.dtype values for '__type__' field."""
    OBJECT, DICT, LIST, FUNCTION, SLICE = range(0, 5)


NoneType = type(None)

PrimitiveType = Union[str, int, float, bool,
                      np.string_, np.bytes_,
                      np.ndarray, np.bool_,
                      np.number, NoneType]

BasicType = Union[
    PrimitiveType,
    Parameter, Parameterized,
    ParamList, Transform, Prior]

TensorType = Union[tf.Variable, tf.Tensor, tf.Operation, tf.data.Iterator]
DictBasicType = Dict[str, BasicType]
ListBasicType = List[BasicType]


class BaseCoder(Contexture, metaclass=abc.ABCMeta):
    """Abstract class for coders."""
    @classmethod
    @abc.abstractmethod
    def support_encoding(cls, item):
        """Decide either encoding is supported for an item or not."""
        pass

    @classmethod
    @abc.abstractmethod
    def support_decoding(cls, item):
        """Decide either decoding is supported for an item or not."""
        pass

    @abc.abstractmethod
    def encode(self, item: Any):
        """Encode input item to structured or plain np.ndarray."""
        pass

    @abc.abstractmethod
    def decode(self, item: np.ndarray):
        """Decode input np.ndarray item back to python type."""
        pass


class PrimitiveTypeCoder(BaseCoder):
    """Coder for primitive python types including numpy array."""

    @classmethod
    def support_encoding(cls, item: PrimitiveType):
        return isinstance(item, cls._types())

    @classmethod
    def support_decoding(cls, item: np.ndarray):
        if _is_numpy_object(item):
            return False
        if _is_nan(item) or _is_str(item):
            return True
        return isinstance(item, cls._types())

    def encode(self, item: PrimitiveType):
        if isinstance(item, str):
            return np.string_(item)
        return numpy_none() if item is None else np.array(item)

    def decode(self, item: np.ndarray):
        if _is_str(item):
            return _convert_to_string(item)
        return None if _is_nan(item) else item

    @classmethod
    def _types(cls):
        return _get_type_args(PrimitiveType)


class TensorFlowCoder(BaseCoder):
    """Coder for TensorFlow tensors and dataset iterators. TensorFlow objects are not
    serialized at all, they are stored as None."""

    @classmethod
    def support_encoding(cls, item):
        supported_types = _get_type_args(TensorType)
        return isinstance(item, supported_types)

    @classmethod
    def support_decoding(cls, _item):
        pass

    def encode(self, item):
        return numpy_none()

    def decode(self, _item):
        return None


class StructCoder(BaseCoder):
    """
    Coder for composite types like List, Dict, Slice, Objects et cetera.
    It defines two abstract methods *encoding_type* and *decoding_type*.
    All structure-like pythonic types are encoded as structured numpy arrays
    with required field '__type__' and other optional fields, which are defined
    by particular implementation.
    """

    @classmethod
    @abc.abstractmethod
    def encoding_type(cls):
        """Non-plain python type for encoding: list, dict or any other structure-like type."""
        pass

    @classmethod
    @abc.abstractmethod
    def decoding_type(cls):
        """Return StructType integer representation of the type which encoded in numpy array."""
        pass

    @classmethod
    def struct(cls, type_name: int, data: np.ndarray, data_dtype: np.dtype = None, shape: Tuple = None):
        """Build structured numpy array.

        :param int type_name: StructType enum converted to integer.
        :param data: encoded data as a numpy array or a structured numpy array.
        :param data_dtype: numpy dtype.
        :param shape: in case when a list of numpy arrays is passed the length (shape) of
            that array is required.

        :return: structured numpy array.
            {
                '__type__': <struct_field_number>,
                '__data__': <numpy_array>
            }
        """

        data_dtype = data.dtype if data_dtype is None else data_dtype
        shape = data.shape if shape is None and data.shape else shape
        data_dtype = [data_dtype]
        if shape:
            data_dtype = [data_dtype[0], shape]
        dtype = np.dtype([type_pattern(), (StructField.DATA.value, *data_dtype)])
        return np.array((type_name, data), dtype=dtype)

    @classmethod
    def support_encoding(cls, item):
        return isinstance(item, cls.encoding_type())

    @classmethod
    def support_decoding(cls, item):
        if not _is_numpy_object(item):
            return False
        fields = item.dtype.fields
        type_field = StructField.TYPE.value
        if not fields or type_field not in fields:
            return False
        return item[type_field] == cls.decoding_type()


class ListCoder(StructCoder):
    """List coder encodes a python list of acceptable types.
    List is encoded into structured numpy array, using dictionary it would
    look like:
    {
        '__type__': StructField.LIST.value,
        '__data__': [<element0>, <element1>, ..., <elementN>]
    }
    Caveat: __data__ is actually a numpy structured array."""

    @classmethod
    def decoding_type(cls):
        return StructType.LIST.value

    @classmethod
    def encoding_type(cls):
        return list

    def encode(self, item: ListBasicType):
        dispatcher = CoderDispatcher(self.context)
        data = [dispatcher.encode(e) for e in item]
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

    def decode(self, item: np.ndarray):
        data = item[StructField.DATA.value]
        if _is_nan(data):
            return []
        if not data.shape:
            keys = sorted(data.dtype.fields.keys())
            data = [data[k] for k in keys]
        dispatcher = CoderDispatcher(self.context)
        return [dispatcher.decode(d) for d in data]


class DictCoder(StructCoder):
    """Dict coder encodes a python dictionary with strings as
    keys and values can be any acceptable pythonic type.
    Encoded dictionary is a numpy structured array and can be
    expressed as dictionary:
    {
        '__type__': StructField.DICT.value,
        '__data__': [(<key0>, <element0>), ..., (<keyN>, <elementN>)]
    },
    Caveat: __data__ is actually a numpy structured array."""
    @classmethod
    def decoding_type(cls):
        return StructType.DICT.value

    @classmethod
    def encoding_type(cls):
        return dict

    def encode(self, item: DictBasicType):
        dispatcher = CoderDispatcher(self.context)
        pre_data = {k : dispatcher.encode(v) for k, v in item.items()}
        if not pre_data:
            data = numpy_none()
        else:
            data_values = [v for _, v in pre_data.items()]
            data_dtype = _list_of_dtypes(pre_data)
            data = np.array(tuple(data_values), dtype=data_dtype)
        return self.struct(self.decoding_type(), data)

    def decode(self, item: np.ndarray):
        data = item[StructField.DATA.value]
        if _is_nan(data):
            return {}
        dispatcher = CoderDispatcher(self.context)
        return {k : dispatcher.decode(data[k]) for k in data.dtype.fields.keys()}


class SliceCoder(StructCoder):
    """Slice coder encodes a python slice structure. Slice itself is a simple
    tuple of three integers.
    {
        '__type__': StructField.SLICE.value,
        '__data__': np.array([<start>, <stop>, <step>])
    }."""

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
        data = item[StructField.DATA.value]
        def try_decode(e):
            return None if _is_nan(e) else int(e)
        return slice(*map(try_decode, data))


class FunctionCoder(StructCoder):
    """Function coder is able to encode only importable functions.
    Lambdas, class methods and static methods as well can not be encoded.
    {
        '__type__': StructField.DICT.value,
        '__module__': <module_name>,
        '__function__': <function_name>
    }."""
    @classmethod
    def decoding_type(cls):
        return StructType.FUNCTION.value

    @classmethod
    def encoding_type(cls):
        return FunctionType

    def encode(self, item):
        dispatcher = CoderDispatcher(self.context)
        name = dispatcher.encode(item.__name__)
        module = dispatcher.encode(item.__module__)
        dtype = np.dtype([type_pattern(),
                          (StructField.MODULE.value, module.dtype),
                          (StructField.FUNCTION.value, name.dtype)])
        return np.array((self.decoding_type(), module, name), dtype=dtype)

    def decode(self, item):
        dispatcher = CoderDispatcher(self.context)
        module = dispatcher.decode(item[StructField.MODULE.value])
        name = dispatcher.decode(item[StructField.FUNCTION.value])
        return _build_type(module, name)


class ObjectCoder(StructCoder):
    """General object coder encodes python importable object into numpy array with
    following dictionary scheme:
    {
        '__type__': StructField.DICT.value,
        '__module__': <module_name>,
        '__class__': <class_name>,
        '__data__': np.ndarray,
        '__extra__': <any_data>
    }."""

    @classmethod
    def decoding_type(cls):
        return StructType.OBJECT.value

    @classmethod
    def encoding_type(cls):
        return object

    @classmethod
    def support_decoding(cls, item: np.ndarray) -> bool:
        if not super().support_decoding(item):
            return False
        dispatcher = CoderDispatcher(BaseContext())
        module = dispatcher.decode(item[StructField.MODULE.value])
        name = dispatcher.decode(item[StructField.CLASS.value])
        item_type = _build_type(module, name)
        return issubclass(item_type, cls.encoding_type())

    def encode(self, item: object) -> np.ndarray:
        name = self._take_class_name(item)
        module = self._take_module_name(item)
        values = self._take_values(item)

        dispatcher = CoderDispatcher(self.context)
        data = dispatcher.encode(values)

        extra = self._take_extras(item)
        extra_data = dispatcher.encode(extra)

        name = dispatcher.encode(item.__class__.__name__)
        module = dispatcher.encode(item.__module__)
        dtype = np.dtype([type_pattern(),
                          (StructField.MODULE.value, module.dtype),
                          (StructField.CLASS.value, name.dtype),
                          (StructField.DATA.value, data.dtype),
                          (StructField.EXTRA.value, extra_data.dtype)])
        return np.array((StructType.OBJECT.value, module, name, data, extra_data), dtype=dtype)

    def decode(self, item: np.ndarray) -> object:
        variables = self._decode_attributes(item)
        return self._decode_object(item, variables)

    def _take_module_name(self, item: object) -> str:
        return item.__class__.__module__

    def _take_class_name(self, item: object) -> str:
        return item.__class__.__name__

    def _take_values(self, item: object) -> DictBasicType:
        return copy(vars(item))

    def _take_extras(self, item: object):
        pass

    def _transform_values(self, _item, values: DictBasicType) -> np.ndarray:
        return CoderDispatcher(self.context).encode(values)

    def _transform_extra(self, _item, extra: bool) -> np.ndarray:
        return CoderDispatcher(self.context).encode(extra)

    def _decode_attributes(self, item: np.ndarray) -> DictBasicType:
        data = item[StructField.DATA.value]
        return CoderDispatcher(self.context).decode(data)

    def _decode_object(self, item: np.ndarray, attributes: DictBasicType) -> object:
        dispatcher = CoderDispatcher(self.context)
        module = dispatcher.decode(item[StructField.MODULE.value])
        name = dispatcher.decode(item[StructField.CLASS.value])
        item_type = _build_type(module, name)
        instance = object.__new__(item_type)
        instance.__dict__ = attributes
        return instance


class TransformCoder(ObjectCoder):
    """Coder for GPflow Transoform objects."""

    @classmethod
    def encoding_type(cls):
        return Transform


class PriorCoder(ObjectCoder):
    """Coder for GPflow Prior objects."""

    @classmethod
    def encoding_type(cls):
        return Prior


class NodeCoder(ObjectCoder):
    """Coder for GPflow Node objects.
    It overrides private `_take_values` method, because nodes have link to the parent node.
    To prevent infinite recursion we setup `_parent` property to None."""

    @classmethod
    def encoding_type(cls):
        return Node

    def _take_values(self, item: Node) -> DictBasicType:
        """Takes snapshot of the object and replaces _parent property value on None to avoid
        infitinite recursion in GPflow tree traversing.

        :param item: GPflow node object.
        :return: dictionary snapshot of the node object."""

        values = super()._take_values(item)
        values['_parent'] = None
        return values


class ParameterCoder(NodeCoder):
    """Coder for GPflow Parameter objects."""

    @classmethod
    def encoding_type(cls):
        return Parameter

    def _take_values(self, item: Parameter) -> DictBasicType:
        """Uses super()._take_values() method, but replaces content of
        the cached value to the value assossiated with context's session.

        :param item: GPflow parameter.
        :return: dictionary snapshot of the parameter object."""

        session = self.context.session
        values = super()._take_values(item)
        cached_value = np.array(item.read_value(session=session))
        values['_value'] = cached_value
        return values

    def _take_extras(self, item: Parameter) -> Optional[bool]:
        """Return either this GPflow objects requires compilation at decoding time.

        :param item: GPflow parameter.
        :return: None or True value.
        """

        index = item.tf_compilation_index()
        if index is not None:
            if item.index == index:
                return True
            _add_index_to_compilations(self.context, index)
        return None

    def _decode_object(self, item: np.ndarray, attributes: DictBasicType) -> Parameter:
        instance = super()._decode_object(item, attributes)
        extra = item[StructField.EXTRA.value]
        extra = CoderDispatcher(self.context).decode(extra)
        if extra and self.context.autocompile:
            instance.compile(session=self.context.session)
        return instance


class ParameterizedCoder(NodeCoder):
    """Coder for GPflow Parameterized objects."""

    @classmethod
    def _encoding_type(cls):
        return Parameterized

    def _take_values(self, item: Parameterized) -> DictBasicType:
        """Uses super()._take_values() method and removes autoflow cache in-place.

        :param item: GPflow parameterized object.
        :return: dictionary snapshot of the parameter object."""

        values = super()._take_values(item)
        values = {k: v for k, v in values.items() if not k.startswith(AutoFlow.__autoflow_prefix__)}
        return values

    def _decode_object(self, item: np.ndarray, attributes: DictBasicType) -> Parameterized:
        instance = super()._decode_object(item, attributes)
        for attr in attributes.values():
            if isinstance(attr, Node):
                setattr(attr, '_parent', instance)
        extra = item[StructField.EXTRA.value]
        extra = CoderDispatcher(self.context).decode(extra)
        if extra and self.context.autocompile:
            instance.compile(session=self.context.session)
        return instance

    def _take_extras(self, item: Parameterized) -> Optional[bool]:
        if _check_index_in_compilations(self.context, item.index):
            return True
        return None


class ParamListCoder(ParameterizedCoder):
    """Coder for GPflow ParamList objects."""
    @classmethod
    def encoding_type(cls):
        return ParamList


class CoderDispatcher(BaseCoder):
    """Dispatcher of coders. Dispatcher uses `support_[encode|decode]` method
    to decide which coder is [encode|decode]ing input value."""

    @classmethod
    def support_decoding(cls, item):
        pass

    @classmethod
    def support_encoding(cls, item):
        pass

    @property
    def coders(self):
        """List of default supported coders. First coder in the list has higher priority."""
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

    def _execute_coder(self, item: Any, coding: str):
        coders = self.context.coders + self.coders
        for coder in coders:
            if coding == 'encode' and coder.support_encoding(item):
                return coder(self.context).encode(item)
            elif coding == 'decode' and coder.support_decoding(item):
                return coder(self.context).decode(item)
        msg = 'Item "{}" has type {} which does not match any coder at saver for {}.'
        raise TypeError(msg.format(item, type(item), coding))

    def encode(self, item: Union[BasicType, ListBasicType, DictBasicType]) -> np.ndarray:
        return self._execute_coder(item, 'encode')

    def decode(self, item: np.ndarray) -> Union[BasicType, ListBasicType, DictBasicType]:
        return self._execute_coder(item, 'decode')


# ====================
# Auxillary functions.
# ====================


def type_pattern():
    return (StructField.TYPE.value, np.uint8)


def empty_array():
    return np.array([], np.uint8)


def numpy_none():
    return np.array(np.nan)


def _add_index_to_compilations(context, index):
    compilations = 'compilations'
    if compilations not in context.shared_data:
        context.shared_data[compilations] = set([])
    context.shared_data[compilations].add(index)


def _check_index_in_compilations(context: BaseContext, index: str):
    """Store compilation flag at specified index in context's shared data."""
    compilations = 'compilations'
    if compilations not in context.shared_data:
        return False
    return index in context.shared_data[compilations]


def _build_type(module_name: str, object_name: str):
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


def _list_of_dtypes(values: Union[DictBasicType, List[Tuple[str, BasicType]]]) -> List[np.dtype]:
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


def _is_shapeless(value: np.ndarray):
    shape = value.shape
    if not shape or (len(shape) == 1 and shape[0] == 0):
        return True
    return False


def _is_numpy_object(value):
    if not isinstance(value, (np.ndarray, np.void)):
        return False
    return value.dtype.type is np.void


def _convert_to_string(value: Union[AnyStr, np.ndarray]):
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = np.string_(value)
    return value.decode('utf-8')


def _get_type_args(union_type):
    if hasattr(union_type, '__args__'):
        return union_type.__args__
    if hasattr(union_type, '__union_params__'):
        return union_type.__union_params__
