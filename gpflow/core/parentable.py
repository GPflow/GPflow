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


class Parentable:
    """
    A very simple class for objects in a tree, where each node contains a
    reference to '_parent'.

    This class can figure out its own name (by seeing what it's called by the
    _parent's __dict__) and also recurse up to the root.
    """

    __index = 0

    def __init__(self, name=None):
        self._parent = None
        self._name = self._define_name(name)

    @property
    def root(self):
        """A reference to the top of the tree, usually a Model instance"""
        if self._parent is None:
            return self
        return self._parent.root

    @property
    def parent(self):
        if self._parent is None:
            return self
        return self._parent

    @property
    def name(self):
        name = self._name.split(sep='/', maxsplit=1)
        if len(name) > 1 and name[0].isdigit():
            return name[1]
        return self._name

    @property
    def hidden_name(self):
        return self._name

    @property
    def full_name(self):
        if self._parent is None:
            return self.name
        return misc.tensor_name(self._parent.full_name, self.name)

    @property
    def hidden_full_name(self):
        """
        This is a unique identifier for a param object within a structure, made
        by concatenating the names through the tree.
        """
        if self._parent is None:
            return self._name
        return misc.tensor_name(self._parent.hidden_full_name, self.name)

    def set_name(self, name=None):
        self._name = self._define_name(name)

    def set_parent(self, parent=None):
        self._parent = parent

    def _reset_name(self):
        if self.hidden_name != self.name:
            self._name = self._define_name(None)

    def _define_name(self, name):
        if name:
            return name
        cls_name = self.__class__.__name__
        index = Parentable._read_index(self)
        return '{index}/{name}'.format(index=index, name=cls_name)

    @classmethod
    def _read_index(cls, obj=None):
        index = cls.__index
        # TODO(@awav): index can grow indefinetly,
        # check boundaries or make another solution.
        cls.__index += 1
        return index
