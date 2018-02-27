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


import sys
import abc
from collections import namedtuple
from copy import copy
from enum import Enum

import numpy as np
import tensorflow as tf

from inspect import isfunction
from ..core import Node, AutoFlow
from ..params import Parameter, Parameterized, ParamList
from ..priors import Prior
from ..transforms import Transform
from .context import Contexture


Struct = namedtuple('Struct', ['module_name',
                               'class_name',
                               'variables',
                               'extra'])

Function = namedtuple('Function', ['module_name', 'function_name'])


class FrameCoding(Enum):
    ENCODE = 0
    DECODE = 1


class BaseFrame(Contexture, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        pass

    @abc.abstractmethod
    def encode(self, item):
        pass

    @abc.abstractmethod
    def decode(self, item):
        pass


class PrimitiveTypeFrame(BaseFrame):
    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        types = (str, int, float, bool, np.ndarray, np.generic, type(None))
        if coding == FrameCoding.ENCODE:
            return isinstance(item, types)
        if isinstance(item, np.void) and item.dtype not in np.ScalarType:
            return False
        return isinstance(item, types)

    def encode(self, item):
        return item
    
    def decode(self, item):
        return item


class ListFrame(BaseFrame):
    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        return isinstance(item, list)

    def encode(self, item):
        factory = FrameFactory(self.context)
        return [factory.encode(e) for e in item]
    
    def decode(self, item):
        factory = FrameFactory(self.context)
        return [factory.decode(e) for e in item]


class DictFrame(BaseFrame):
    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        return isinstance(item, dict)

    def encode(self, item):
        factory = FrameFactory(self.context)
        # TODO: dictionary key must be a string
        return {k : factory.encode(v) for k, v in item.items()}
    
    def decode(self, item):
        factory = FrameFactory(self.context)
        return {k : factory.decode(v) for k, v in item.items()}


class TensorFlowFrame(BaseFrame):
    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        supported_types = (tf.Variable, tf.Tensor, tf.Operation, tf.data.Iterator)
        return isinstance(item, supported_types)
    
    def encode(self, item):
        return None

    def decode(self, _item):
        return None


class SliceFrame(BaseFrame):
    _slice_numpy_dtype = np.dtype([('start', float), ('stop', float), ('step', float)])

    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        if coding == FrameCoding.ENCODE:
            return isinstance(item, slice)
        if not isinstance(item, np.void):
            return False
        return np.issubdtype(item.dtype, cls._slice_numpy_dtype)
    
    def encode(self, item):
        slice_tuple = (item.start, item.stop, item.step)
        return np.array(slice_tuple, dtype=self._slice_numpy_dtype)

    def decode(self, item):
        def element_of_slice(e):
            if np.isnan(e):
                return None
            return int(e)
        return slice(*map(element_of_slice, item.item()))


def _real_object_type(module_name, object_name):
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


class FunctionFrame(BaseFrame):
    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        if coding == FrameCoding.ENCODE:
            return isfunction(item)
        return isinstance(item, Function)
    
    def encode(self, item):
        return Function(module_name=item.__module__, function_name=item.__name__)
    
    def decode(self, item):
        return _real_object_type(item.module_name, item.function_name)


class ObjectFrame(BaseFrame):
    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        if coding == FrameCoding.ENCODE:
            return cls._support_encoding(item)
        return cls._support_decoding(item)
    
    @classmethod
    def _encoding_type(cls):
        return object

    @classmethod
    def _support_encoding(cls, item):
        return isinstance(item, cls._encoding_type())
    
    @classmethod
    def _support_decoding(cls, item):
        if not isinstance(item, Struct):
            return False
        encoding_type = cls._encoding_type()
        if encoding_type is not object:
            item_type = _real_object_type(item.module_name, item.class_name)
            return issubclass(item_type, encoding_type)
        return True
    
    def encode(self, item):
        module_name = self._take_module_name(item)
        class_name = self._take_class_name(item)
        values = self._take_values(item)
        values = self._transform_values(item, values)
        extra = self._take_extras(item)
        return Struct(module_name, class_name, values, extra)
    
    def decode(self, item):
        variables = self._create_attributes(item)
        return self._create_object(item, variables)
    
    def _take_module_name(self, item):
        return item.__class__.__module__

    def _take_class_name(self, item):
        return item.__class__.__name__
    
    def _take_values(self, item):
        return copy(vars(item))
    
    def _take_extras(self, item):
        pass
    
    def _transform_values(self, _item, values):
        factory = FrameFactory(self.context)
        return {k : factory.encode(v) for (k, v) in values.items()}
    
    def _create_attributes(self, item):
        factory = FrameFactory(self.context)
        return {factory.decode(key) : factory.decode(value) for key, value in item.variables.items()}
    
    def _create_object(self, item, attributes):
        item_type = _real_object_type(item.module_name, item.class_name)
        instance = object.__new__(item_type)
        instance.__dict__ = attributes
        return instance


class TransformFrame(ObjectFrame):
    @classmethod
    def _encoding_type(cls):
        return Transform


class PriorFrame(ObjectFrame):
    @classmethod
    def _encoding_type(cls):
        return Prior


class ParamListFrame(ObjectFrame):
    @classmethod
    def _encoding_type(cls):
        return ParamList

class NodeFrame(ObjectFrame):
    @classmethod
    def _encoding_type(cls):
        return Node
    
    def _take_values(self, item):
        values = super()._take_values(item)
        values['_parent'] = None
        return values


class ParameterFrame(NodeFrame):
    @classmethod
    def _encoding_type(cls):
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
    
    def _create_object(self, item, attributes):
        instance = super()._create_object(item, attributes)
        if item.extra and self.context.autocompile:
            instance.compile(session=self.context.session)
        return instance

class ParameterizedFrame(NodeFrame):
    @classmethod
    def _encoding_type(cls):
        return Parameterized
    
    def _take_values(self, item):
        values = super()._take_values(item)
        values = {k : v for k, v in values.items() if not k.startswith(AutoFlow.__autoflow_prefix__)}
        return values

    def _create_object(self, item, attributes):
        instance = super()._create_object(item, attributes)
        for attr in attributes.values():
            if isinstance(attr, Node):
                setattr(attr, '_parent', instance)
        if item.extra and self.context.autocompile:
            instance.compile(session=self.context.session)
        return instance

    def _take_extras(self, item):
        if _check_index_in_compilations(self.context, item.index):
            return True
        return None

class FrameFactory(BaseFrame):
    @classmethod
    def support(cls, item, coding=FrameCoding.ENCODE):
        pass

    @property
    def frames(self):
        return (PrimitiveTypeFrame,
                SliceFrame,
                TensorFlowFrame,
                FunctionFrame,
                ListFrame,
                DictFrame,
                ParameterFrame,
                ParamListFrame,
                ParameterizedFrame,
                TransformFrame,
                PriorFrame)
    
    def find_supported_frame(self, item, coding):
        frames = self.context.frames + self.frames
        for frame in frames:
            if frame.support(item, coding=coding):
                return frame
        msg = 'Item "{}" has type {} which does not match any frame at saver for {}.'
        raise TypeError(msg.format(item, type(item), coding))

    def encode(self, item):
        return self.__execute_operation(item, FrameCoding.ENCODE)
    
    def decode(self, item):
        return self.__execute_operation(item, FrameCoding.DECODE)

    def __execute_operation(self, item, coding):
        def run_method(e):
            if coding == FrameCoding.ENCODE:
                return e.encode(item)
            return e.decode(item)
        frame = self.find_supported_frame(item, coding)
        return run_method(frame(self.context))


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
