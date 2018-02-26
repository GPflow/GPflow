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


import tensorflow as tf

from .. import get_default_session


class Contexture:
    def __init__(self, context):
        self._context = context
    
    @property
    def context(self):
        return self._context


class BaseContext:
    def __init__(self, frames=None, serializer=None, autocompile=True):
        self.autocompile = autocompile
        self.session = tf.get_default_session() or get_default_session()
        self._custom_frames = () if frames is None else tuple(frames)
        self._shared_data = {}
    
    @property
    def frames(self):
        return self._custom_frames

    @property
    def shared_data(self):
        return self._shared_data
