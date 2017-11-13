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
import enum
import inspect


# TODO(@awav): Introducing global variable is not best idea for managing compilation, but
# necessary for context manager support.
# Inspect approach works well, except that I have a concern that it can be slow when
# nesting is too deep and it doesn't work well with context managers, because the code which
# is run inside `with` statement doesn't have an access to the frame which caused it.

class AutoBuildStatus(enum.Enum):
    """
    This enum class is used for safe marking current status for global and local
    autobuild settings. It must never be used outside of the project.
    """
    __autobuild_enabled_global__ = True

    BUILD = 1
    IGNORE = 2
    FOLLOW = 3


class AutoBuild(abc.ABCMeta):
    """
    AutoBuild meta class is used for changing an initializing behaviour at its descendants.
    Whenever an object defined with this metaclass is created - the `build` and `initialize`
    methods of ICompilable interface are called immediately after object's __init__ function.

    It also modifies input dictionary arguments for any class which uses
    AutoBuild as metaclass. It adds `autobuild` option, using it you can control either
    ICompilable object builds itself at instantiation or delay it, so that you could run
    compilation later.

    For example a class defined without AutoBuild metaclass will raise TypeError. But if you
    defined a class with AutoBuild metaclass, then it modifies the class __init__ method and
    adds `autobuild` option to any successor of that class.

    ```
    class A(metaclass=AutoBuild):
        def __init__(self):
            pass

    class B():
        def __init__(self):
            pass

    a = A(autobuild=False) # works fine, even when __init__ arguemnt list is empty.
    b = B(autobuild=False) # raises TypeError exception.
    ```
    """

    _autobuild_arg = 'autobuild'

    def __new__(mcs, name, bases, namespace, **kwargs):
        new_cls = super(AutoBuild, mcs).__new__(mcs, name, bases, namespace, **kwargs)
        origin_init = new_cls.__init__
        def __init__(self, *args, **kwargs):
            """
            The `kwargs` may or may not contain 'autobuild' option. This option is
            inherited implicitly by all classes.
            """
            autobuild = kwargs.pop(AutoBuild._autobuild_arg, True)
            __execute_autobuild__ = AutoBuildStatus.BUILD if autobuild else AutoBuildStatus.IGNORE
            tag = '__execute_autobuild__'
            frame = inspect.currentframe().f_back
            while autobuild and frame:
                if isinstance(frame.f_locals.get(tag, None), AutoBuildStatus):
                    __execute_autobuild__ = AutoBuildStatus.FOLLOW
                    break
                frame = frame.f_back
            origin_init(self, *args, **kwargs)
            autobuild_on = __execute_autobuild__ == AutoBuildStatus.BUILD
            global_autobuild_on = AutoBuildStatus.__autobuild_enabled_global__
            if autobuild_on and global_autobuild_on:
                self.build()
                self.initialize(force=True)
        setattr(new_cls, '__init__', __init__)
        return new_cls


class Build(enum.Enum):
    """
    ICompilable object status.
    ICompilable object can be built within one and only one graph, therefore this status
    express either object was built using particular graph. NOT_COMPATIBLE_GRAPH is
    a special case of status, which shows that compilable object embedded in a graph,
    but the user is checking status for different one.
    """

    YES = 1
    NO = 0  # pylint: disable=C0103
    NOT_COMPATIBLE_GRAPH = None


class ICompilable(metaclass=AutoBuild):
    @abc.abstractproperty
    def graph(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def feeds(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def initializables(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def initializable_feeds(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def build(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, session=None, force=False):
        raise NotImplementedError()

    @abc.abstractmethod
    def compile(self, session=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def is_built(self, graph):
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _build(self):
        raise NotImplementedError()
