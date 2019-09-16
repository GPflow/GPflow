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


from .coders import CoderDispatcher
from .context import BaseContext
from .serializers import HDF5Serializer


class SaverContext(BaseContext):
    def __init__(self, version=None, serializer=None, **kwargs):
        super().__init__(**kwargs)
        self.serializer = HDF5Serializer if serializer is None else serializer


class Saver:
    def save(self, pathname_or_file_like, target, context=None):
        context = Saver.__get_context(context)
        encoded_target = CoderDispatcher(context).encode(target)
        context.serializer(context).dump(pathname_or_file_like, encoded_target)

    def load(self, pathname_or_file_like, context=None):
        context = Saver.__get_context(context)
        encoded_target = context.serializer(context).load(pathname_or_file_like)
        return CoderDispatcher(context).decode(encoded_target)

    @staticmethod
    def __get_context(context):
        if context is None:
            context = SaverContext()
        if not isinstance(context, SaverContext):
            raise TypeError('The context must be instance of "SaverContext" class.')
        return context
