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
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf

from .context import BaseContext
from .serializers import HDF5Serializer
from .frames import FrameFactory


class SaverContext(BaseContext):
    def __init__(self, session=None, version=None, serializer=None, **kwargs):
        super().__init__(**kwargs)
        if session is not None:
            self.session = session
        self.serializer = HDF5Serializer if serializer is None else serializer


class Saver:
    def save(self, pathname, target, context=None):
        context = Saver.__get_context(context)
        encoded_target = FrameFactory(context).encode(target)
        context.serializer(context).dump(pathname, encoded_target)

    def load(self, pathname, context=None):
        context = Saver.__get_context(context)
        encoded_target = context.serializer(context).load(pathname)
        return FrameFactory(context).decode(encoded_target)

    @staticmethod
    def __get_context(context):
        if context is None:
            context = SaverContext()
        if not isinstance(context, SaverContext):
            raise ValueError('The context must be instance of "SaverContext" class.')
        return context
