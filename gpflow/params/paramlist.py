# Copyright 2016 James Hensman, Mark van der Wilk,
#                Valentine Svensson, alexggmatthews,
#                PabloLeon, fujiisoup
# Copyright 2017 Artem Artemev @awav
#
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


from __future__ import absolute_import

from ..core.tensor_converter import TensorConverter
from ..core.errors import GPflowError
from ..core.compilable import Build

from .parameter import Parameter
from .parameterized import Parameterized


class ParamList(Parameterized):
    """
    ParamList is special case of parameterized object. It implements different
    access pattern for its childres. Instead saving node like objects as attributes
    it keeps them in the list, providing indexed access and acts as simple list.

    :param list_of_params: list of node like objects.
    :param trainable: Boolean flag. It indicates whether children parameters
        should be trainable or not.
    :param name: ParamList name.
    """

    def __init__(self, list_of_params, trainable=True, name=None):
        super(ParamList, self).__init__(name=None)
        if not isinstance(list_of_params, list):
            raise ValueError('Not acceptable argument type at list_of_params.')
        self._list = [self._valid_list_input(item, trainable)
                      for item in list_of_params]
        for index, item in enumerate(self._list):
            self._set_param(index, item)

    @property
    def params(self):
        for item in self._list:
            yield item

    def append(self, item):
        if not isinstance(item, Parameter):
            raise ValueError(
                'Non parameter type cannot be appended to the list.')
        length = self.__len__()
        item.set_parent(self)
        item.set_name(self._item_name(length))
        self._list.append(item)

    def _get_param(self, name):
        return self._list[name]

    def _set_param(self, name, value):
        self._list[name] = value
        value.set_parent(self)
        value.set_name(self._item_name(name))

    def _item_name(self, index):
        return '{name}{index}'.format(name='item', index=index)

    def _valid_list_input(self, value, trainable):
        if not Parameterized._is_param_like(value):
            try:
                return Parameter(value, trainable=trainable)
            except ValueError:
                raise ValueError(
                    'A list item must be either parameter, '
                    'tensorflow variable, an array or a scalar.')
        return value

    def __len__(self):
        return len(self._list)

    def __getitem__(self, key):
        param = self._get_param(key)
        if TensorConverter.tensor_mode(self) and isinstance(param, Parameter):
            return Parameterized._tensor_mode_parameter(param)
        return param

    def __setitem__(self, index, value):
        if not isinstance(value, Parameter):
            raise ValueError(
                'Non parameter type cannot be assigned to the list.')
        if not self.empty and self.is_built_coherence(value.graph) is Build.YES:
            raise GPflowError(
                'ParamList is compiled and items are not modifiable.')
        self._update_param_attribute(index, value)
