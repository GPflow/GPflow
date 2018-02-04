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
        self._list = []
        for value in list_of_params:
            item = self._valid_list_input(value, trainable)
            self.append(item)

    @property
    def children(self):
        return {str(i): v for i, v in enumerate(self._list)}

    def store_child(self, index, child):
        if index == len(self):
            self._list.append(child)
        else:
            self._list[index] = child

    @property
    def params(self):
        for item in self._list:
            yield item

    def append(self, item):
        value = self._valid_list_input(item, self.trainable)
        index = len(self)
        self._set_node(index, value)

    def _get_node(self, name):
        return self._list[name]

    def _replace_node(self, index, old, new):
        old.set_parent()
        self.set_child(index, new)

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

    def __getitem__(self, index):
        e = self._get_node(index)
        if TensorConverter.tensor_mode(self) and isinstance(e, Parameter):
            return Parameterized._tensor_mode_parameter(e)
        return e

    def __setitem__(self, index, value):
        if not isinstance(value, Parameter):
            raise ValueError(
                'Non parameter type cannot be assigned to the list.')
        if not self.empty and self.is_built_coherence(value.graph) is Build.YES:
            raise GPflowError(
                'ParamList is compiled and items are not modifiable.')
        self._update_node(index, value)
