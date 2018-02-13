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

    __global_index = 0

    def __init__(self, name=None):
        self._parent = None
        index = None
        if name is None:
            name = self._define_name(name)
            index = self._gen_index()
        self._name = name
        self._index = index

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
    
    @property
    def full_name(self):
        """
        Backward compatibility with previous versions.
        WARNING: WILL BE DEPRICATED SOON.
        """
        return self.pathname

    @property
    def index(self):
        return self._index

    @abc.abstractproperty
    def children(self):
        """
        Abstract property for getting pairs of names and children, respectively.

        :return: Dictionary where keys are names and values are Parentable objects.
        """
        pass

    @abc.abstractmethod
    def store_child(self, name, child):
        """
        Abstract method for saving association between child and its name in
        parent's specific storage.
        """
        pass


    @abc.abstractmethod
    def remove_child(self, name, child):
        """
        Abstract method for removing an association between child and its name in
        parent's specific storage.
        """
        pass

    @property
    def descendants(self):
        """
        Scans full list of node descendants.

        :return: Generator of nodes.
        """
        children = self.children
        if children is not None:
            for child in children.values():
                yield from child.descendants
                yield child
    
    def contains(self, node):
        """
        Checks either node already exist somewhere among descendants.

        :return: Boolean value.
        """
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

        :param name: Child name.
        :param child: Parentable object.
        """
        if not isinstance(child, Parentable):
            raise ValueError('Parentable child object expected, not {child}'.format(child=child))
        child.set_parent(self)
        self.store_child(name, child)
    
    def unset_child(self, name, child):
        """
        Untie child from parent.

        :param name: Child name.
        :param child: Parentable object.
        """
        if name not in self.children or self.children[name] is not child:
            msg = 'Child {child} with name "{name}" is not found'
            raise ValueError(msg.format(child=child, name=name))
        child.set_parent(None)
        self.remove_child(name, child)


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
    
    def reset_name(self, name=None):
        self._name = self._define_name(name=name)
        self._index = self._gen_index()

    def _define_name(self, name=None):
        if name is not None:
            return name
        cls_name = self.__class__.__name__
        return cls_name

    def _gen_index(self):
        internal_index = Parentable._read_index()
        uuid_index = str(uuid.uuid4())[:8]
        pattern = '{uuid_index}-{internal_index}'
        return pattern.format(uuid_index=uuid_index, internal_index=internal_index)

    @classmethod
    def _read_index(cls):
        """
        Reads index number and increments it. No thread-safety guarantees.

        # TODO(@awav): index can grow indefinetly,
        # check boundaries or make another solution.
        """
        index = cls.__global_index
        cls.__global_index += 1
        return index
