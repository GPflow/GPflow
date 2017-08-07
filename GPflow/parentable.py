
from __future__ import absolute_import


class Parentable(object):
    """
    A very simple class for objects in a tree, where each node contains a
    reference to '_parent'.

    This class can figure out its own name (by seeing what it's called by the
    _parent's __dict__) and also recurse up to the highest_parent.
    """

    def __init__(self):
        self._parent = None

    @property
    def highest_parent(self):
        """A reference to the top of the tree, usually a Model instance"""
        if self._parent is None:
            return self
        else:
            return self._parent.highest_parent

    @property
    def name_dict(self):
        return self.__dict__


    @property
    def name(self):
        """An automatically generated name, given by the reference of the _parent to this instance"""
        if self._parent is None:
            return 'unnamed'
        # NB delinked this parent class from knowing about child implementations,
        # which generally is probably a good idea. However, before a param list could contain two members of the same
        # class and only return the first one (due to index). Now it will just fail. I think this is probably more
        # sensible. But need to ensure others don't disagree
        #TODO: discuss in review
        matches = [key for key, value in self._parent.name_dict.items()
                   if value is self]
        if len(matches) == 0:
            raise ValueError("mis-specified parent. This Param's\
                             _parent does not contain a reference to it.")
        if len(matches) > 1:
            raise ValueError("This Param appears to be doubly\
                             referenced by a parent")
        return matches[0]

    @property
    def long_name(self):
        """
        This is a unique identifier for a param object within a structure, made
        by concatenating the names through the tree.
        """
        if self._parent is None:
            return self.name
        return self._parent.long_name + '.' + self.name

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_parent')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._parent = None
