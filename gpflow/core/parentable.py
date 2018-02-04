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

import abc
import uuid

from .. import misc


class Parentable:
    """
    Very simple class for organizing GPflow objects in a tree.
    Each node contains a reference to the its parent. Links to children
    established implicitly via python object attributes.

    The Parentable class stores static variable which is used for assigning unique
    prefix name identification for each newly created object inherited from Parentable.

    Name is not required at `Parentable` object creation.
    But in this case, unique index identifier will be attached to the class name.

    :param name: String name.
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
        `parent.pathname + parent.childname` and stop condition is root's
        name.

        For example, the pathname of an instance with the two parents may look
        like `parent0/parent1/childname_at_parent1`.
        Top parent's name equals its original name `parent0.name == parent0.pathname`.
        """
        if self.parent is self:
            return self.name
        parent = self._parent
        return misc.tensor_name(parent.pathname, parent.childname(self))

    @abc.abstractproperty
    def children(self):
        """
        :return: Dictionary where keys are names and values are Parentable objects.
        """
        pass

    @abc.abstractmethod
    def store_child(self, name, child):
        """

        """
        pass

    @property
    def descendants(self):
        children = self.children
        if children is not None:
            for child in children.values():
                yield from child.descendants
                yield child
    
    def contains(self, node):
        return node in list(self.descendants)

    def childname(self, child):
        if not isinstance(child, Parentable):
            raise ValueError('Parentable object expected, {child} passed.'.format(child=child))
        name = [ch[0] for ch in self.children.items() if ch[1] is child]
        if not name:
            raise KeyError('Parent {parent} does not have child {child}'.format(parent=self, child=child))
        return name[0]

    def set_child(self, name, child):
        """
        Set child.

        :param child: Parentable object.
        :param name: String attribute name.

        """
        if not isinstance(child, Parentable):
            raise ValueError('Parentable child object expected, not {child}'.format(child=child))
        child.set_parent(self)
        self.store_child(name, child)

    def set_parent(self, parent=None):
        """
        Set parent.

        :param parent: Parentable object.
        :raises ValueError: Self-reference object passed.
        :raises ValueError: Non-Parentable object passed.
        """
        if parent is not None:
            if not isinstance(parent, Parentable):
                raise ValueError('Parent object must implement Parentable interface.')
            if parent is self or parent.contains(self):
                raise ValueError('Self references are not allowed.')
        self._parent = parent if parent is not None else None

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
