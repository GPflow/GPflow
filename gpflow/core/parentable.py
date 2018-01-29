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

import uuid

from .. import misc


class Parentable:
    """
    Very simple class for organizing GPflow objects in a tree.
    Each node contains a reference to the its parent. Links to children
    established implicitly via attributes.

    The Parentable class stores static variable which is used for assigning unique
    prefix name identificator for each newly created object inherited from Parentable.

    Name is not required at `Parentable` object creation.
    But in this case, unique index identifier will be attached to the hidden name
    (internal representation name). The index is cut off, once node is placed under
    another parent node.

    :param name: String parentable nodes name.
    """

    __index = 0

    def __init__(self, name=None):
        self._parent = None
        self._name = self._define_name(name)

    @property
    def root(self):
        """
        Top of the parentable tree.
        :return: Reference to top parentable object.
        """
        if self._parent is None:
            return self
        return self._parent.root

    @property
    def parent(self):
        """
        Parent for this node.
        :return: Reference to parent object.
        """
        if self._parent is None:
            return self
        return self._parent

    @property
    def name(self):
        """
        The name assigned to node at creation time. It can be referred also
        as original name.

        :return: Given name.
        """
        return self._name

    @property
    def pathname(self):
        """
        Path name is a recursive representation parent path name plus the name
        which was assigned to this object by its parent. In other words, it is
        stack of parent name where top is always parent's original name:
        `parent.pathname + parent.child_name` and stop condition is root's
        name.

        For example, the pathname of an instance with the two parents may look
        like `parent0/parent1/child_name_at_parent1`.
        Top parent's name equals its original name `parent0.name == parent0.pathname`.
        """
        if self._parent is None:
            return self.name
        return misc.tensor_name(self._parent.full_name, self.name)

    def set_name(self, name=None):
        self._name = self._define_name(name) if name is None else name

    def set_parent(self, parent=None):
        self._parent = parent

    def _reset_name(self):
        self._name = self._define_name()

    def _define_name(self, name=None):
        if name is not None:
            if not isinstance(name, str):
                raise ValueError('Name must be a string.')
            return name
        cls_name = self.__class__.__name__
        index = Parentable._read_index()
        rnd_index = str(uuid.uuid4())[:8] + str(index)
        return '{name}-{rnd_index}{index}'.format(name=cls_name, rnd_index=rnd_index, index=index)

    @classmethod
    def _read_index(cls):
        """
        Reads index number and increments it. No thread-safety guarantees.

        # TODO(@awav): index can grow indefinetly,
        # check boundaries or make another solution.
        """
        index = cls.__index
        cls.__index += 1
        return index
