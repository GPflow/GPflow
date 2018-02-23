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

import numpy as np
import tensorflow as tf

from ..core import Node
from ..params import Parameter, Parameterized, ParamList
from ..priors import Prior
from ..transforms import Transform
from .context import Contexture


Struct = namedtuple('Struct', ['module_name',
                               'class_name',
                               'variables',
                               'extra'])


class BaseFrame(Contexture, metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def supported_types():
        pass

    @abc.abstractmethod
    def encode(self, item):
        pass

    @abc.abstractmethod
    def decode(self, item):
        pass

    def _check_encode_type(self, item):
        self._check_type(item, 'encoding')
    
    def _check_decode_type(self, item):
        self._check_type(item, 'decoding')
    
    def _check_type(self, item, message):
        if not isinstance(item, self.supported_types()):
            msg = '{class_name} does not support type {item_type} for {message}.'
            msg = msg.format(
                class_name=self.__class__.__name__,
                item_type=type(item),
                message=message)
            raise ValueError(msg)


class PrimitiveTypeFrame(BaseFrame):
    @staticmethod
    def supported_types():
        return (str, int, float, bool, np.ndarray, type(None), np.generic)

    def encode(self, item):
        return item
    
    def decode(self, item):
        return item


class ListFrame(BaseFrame):
    @staticmethod
    def supported_types():
        return list

    def encode(self, item):
        factory = FrameFactory(self.context)
        return [factory.encode(e) for e in item]
    
    def decode(self, item):
        factory = FrameFactory(self.context)
        return [factory.decode(e) for e in item]


class DictFrame(BaseFrame):
    @staticmethod
    def supported_types():
        return (dict)

    def encode(self, item):
        factory = FrameFactory(self.context)
        # TODO: dictionary key must be a string
        return {k : factory.encode(v) for k, v in item.items()}
    
    def decode(self, item):
        factory = FrameFactory(self.context)
        return {k : factory.decode(v) for k, v in item.items()}


class TensorFlowFrame(BaseFrame):
    @staticmethod
    def supported_types():
        return (tf.Variable, tf.Tensor, tf.Operation, tf.data.Iterator)
    
    def encode(self, item):
        return None

    def decode(self, _item):
        return None


def _real_object_struct_type(c):
    try:
        __import__(c.module_name)
    except ModuleNotFoundError:
        raise RuntimeError('TODO')
    module = sys.modules[c.module_name]
    try:
        return module.__dict__[c.class_name]
    except KeyError:
        raise RuntimeError('TODO')


class ObjectFrame(BaseFrame):
    @staticmethod
    def supported_types():
        return object
    
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
        item_type = _real_object_struct_type(item)
        instance = object.__new__(item_type)
        instance.__dict__ = attributes
        return instance


class TransformFrame(ObjectFrame):
    @staticmethod
    def supported_types():
        return Transform


class PriorFrame(ObjectFrame):
    @staticmethod
    def supported_types():
        return Prior


class ParamListFrame(ObjectFrame):
    @staticmethod
    def supported_types():
        return ParamList

class NodeFrame(ObjectFrame):
    @staticmethod
    def supported_types():
        return Node
    
    def _take_values(self, item):
        values = super()._take_values(item)
        values['_parent'] = None
        return values


class ParameterFrame(NodeFrame):
    @staticmethod
    def supported_types():
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
        if item.extra:
            instance.compile(session=self.context.session)
        return instance

class ParameterizedFrame(NodeFrame):
    @staticmethod
    def supported_types():
        return Parameterized

    def _create_object(self, item, attributes):
        instance = super()._create_object(item, attributes)
        for attr in attributes.values():
            if isinstance(attr, Node):
                setattr(attr, '_parent', instance)
        if item.extra:
            instance.compile(session=self.context.session)
        return instance

    def _take_extras(self, item):
        if _check_index_in_compilations(self.context, item.index):
            return True
        return None

class FrameFactory(BaseFrame):
    @staticmethod
    def supported_types():
        return ()

    @staticmethod
    def frames():
        return (PrimitiveTypeFrame,
                TensorFlowFrame,
                ListFrame,
                DictFrame,
                ParameterFrame,
                ParamListFrame,
                ParameterizedFrame,
                TransformFrame,
                PriorFrame)
    
    def find_supported_frame(self, item):
        if isinstance(item, Struct):
            item_type = _real_object_struct_type(item)
        else:
            item_type = type(item)
        frames = self.context.frames + self.frames()
        for e in frames:
            if issubclass(item_type, e.supported_types()):
                return e
        msg = 'Item "{}" has type {} which does not match any frame at saver.'
        raise TypeError(msg.format(item, item_type))

    def encode(self, item):
        return self.__execute_operation(item, 'encode')
    
    def decode(self, item):
        return self.__execute_operation(item, 'decode')

    def __execute_operation(self, item, method):
        def run_method(e):
            return getattr(e, method)(item)
        frame = self.find_supported_frame(item)
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
