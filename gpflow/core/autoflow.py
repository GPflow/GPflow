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


from .. import misc


class AutoFlow:
    """
    AutoFlow is responsible for managing tensor storages for different session runs
    of GPflow objects.
    """

    __autoflow_prefix__ = '_autoflow_'

    @classmethod
    def get_autoflow(cls, obj, name):
        """
        Extracts from an object existing dictionary with tensors specified by name.
        If there is no such object then new one will be created. Intenally, it appends
        autoflow prefix to the name and saves it as an attribute.

        :param obj: target GPflow object.
        :param name: unique part of autoflow attribute's name.

        :raises: ValueError exception if `name` is not a string.
        """
        if not isinstance(name, str):
            raise ValueError('Name must be string.')
        prefix = cls.__autoflow_prefix__
        autoflow_name = prefix + name
        store = misc.get_attribute(obj, autoflow_name, allow_fail=True, default={})
        if not store:
            setattr(obj, autoflow_name, store)
        return store

    @classmethod
    def clear_autoflow(cls, obj, name=None):
        """
        Clear autoflow's tensor storage.

        :param obj: target GPflow object.
        :param name: accepts either string value which is unique part of
            an internal attribute name or None value. When None value is passed all
            storages will be cleared, in other words it clears everything with common
            autoflow prefix.

        :raises: ValueError exception if `name` is not a string.
        """
        if name is not None and not isinstance(name, str):
            raise ValueError('Name must be a string.')
        prefix = cls.__autoflow_prefix__
        if name:
            prefix = "" if name.startswith(prefix) else prefix
            delattr(obj, prefix + name)
        else:
            keys = [attr for attr in obj.__dict__ if attr.startswith(prefix)]
            for key in keys:
                delattr(obj, key)
