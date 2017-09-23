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


class AutoFlow:
    __autoflow_prefix__ = '_autoflow_'

    @classmethod
    def get_autoflow(cls, obj, name):
        if not isinstance(name, str):
            raise ValueError('Name must be string.')
        prefix = cls.__autoflow_prefix__
        autoflow_name = prefix + name
        store = get_attribute(obj, autoflow_name, {})
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
