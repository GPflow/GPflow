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
import decorator


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


class AutobuildFunctionMaker(decorator.FunctionMaker):
    """
    This overwrites the __init__ of decorator.FunctionMaker to insert the autobuild keyword argument.
    """

    def __init__(self, func=None, name=None, signature=None,
                 defaults=None, doc=None, module=None, funcdict=None):
        self.shortsignature = signature
        if func:
            # func can be a class or a callable, but not an instance method
            self.name = func.__name__
            if self.name == '<lambda>':  # small hack for lambda functions
                self.name = '_lambda_'
            self.doc = func.__doc__
            self.module = func.__module__
            if inspect.isfunction(func):
                argspec = inspect.getfullargspec(func)
                self.annotations = getattr(func, '__annotations__', {})
                for a in ('args', 'varargs', 'varkw', 'defaults', 'kwonlyargs',
                          'kwonlydefaults'):
                    setattr(self, a, getattr(argspec, a))
                ##### BEGIN OF CHANGES
                # The following four lines are manually added for GPflow:
                self.kwonlyargs.append('autobuild')
                if self.kwonlydefaults is None:
                    self.kwonlydefaults = {}
                self.kwonlydefaults['autobuild'] = True
                ##### END OF CHANGES
                for i, arg in enumerate(self.args):
                    setattr(self, 'arg%d' % i, arg)
                allargs = list(self.args)
                allshortargs = list(self.args)
                if self.varargs:
                    allargs.append('*' + self.varargs)
                    allshortargs.append('*' + self.varargs)
                elif self.kwonlyargs:
                    allargs.append('*')  # single star syntax
                for a in self.kwonlyargs:
                    allargs.append('%s=None' % a)
                    allshortargs.append('%s=%s' % (a, a))
                if self.varkw:
                    allargs.append('**' + self.varkw)
                    allshortargs.append('**' + self.varkw)
                self.signature = ', '.join(allargs)
                self.shortsignature = ', '.join(allshortargs)
                self.dict = func.__dict__.copy()
        # func=None happens when decorating a caller
        if name:
            self.name = name
        if signature is not None:
            self.signature = signature
        if defaults:
            self.defaults = defaults
        if doc:
            self.doc = doc
        if module:
            self.module = module
        if funcdict:
            self.dict = funcdict
        # check existence required attributes
        assert hasattr(self, 'name')
        if not hasattr(self, 'signature'):
            raise TypeError('You are decorating a non function: %s' % func)

def autobuild_decorate(func, caller):
    """
    autobuild_decorate(func, caller) decorates a function using a caller.
    Allows for an extra `autobuild` keyword arg.
    The only difference to decorator.decorate() is to use our custom AutobuildFunctionMaker instead.
    """
    evaldict = dict(_call_=caller, _func_=func)
    fun = AutobuildFunctionMaker.create(
        func, "return _call_(_func_, %(shortsignature)s)",
        evaldict, __wrapped__=func)
    if hasattr(func, '__qualname__'):
        fun.__qualname__ = func.__qualname__
    return fun

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

    a = A(autobuild=False) # works fine, even when __init__ argument list is empty.
    b = B(autobuild=False) # raises TypeError exception.
    ```
    """

    _autobuild_arg = 'autobuild'

    def __new__(mcs, name, bases, namespace, **kwargs):
        new_cls = super(AutoBuild, mcs).__new__(mcs, name, bases, namespace, **kwargs)
        if getattr(new_cls.__init__, '_autobuild_wrapped_', False):
            # The decorator hack to make the autobuild keyword argument work does not play nicely
            # with recursive wrapping. If we already wrapped this __init__ (because a class does
            # not overwrite its superclass's __init__), just let it be.
            return new_cls

        def init_wrapper(origin_init, self, *args, **kwargs):
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

        # Keep the original __init__'s signature and documentation:
        __init__ = autobuild_decorate(new_cls.__init__, init_wrapper)
        __init__.__doc__ = new_cls.__init__.__doc__

        # Track that this specific method already got wrapped:
        __init__._autobuild_wrapped_ = True

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
    def __init__(self):
        # We need an empty __init__ so that the AutoBuild decorator magic works.
        # Otherwise, the type of __init__ would be wrapper_descriptor, not a function.
        pass

    @abc.abstractproperty
    def graph(self):
        """
        TensorFlow graph property.

        :return: tf.Graph which was used during building.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def feeds(self):
        """
        TensorFlow feed dictionary for passing to tf.Session.run()

        :return: TensorFlow feed dictionary or None.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def initializables(self):
        """
        List of TensorFlow tensors which must be initialized.
        This list is necessary for successfull _initialize_ call.

        :return: List of TensorFlow variables, data iterators or both,
            which are capable to be initialized.

        """
        raise NotImplementedError()

    @abc.abstractproperty
    def initializable_feeds(self):
        """
        Feed dictionary which will be used along with `initializables` list
        at `initialize` function.

        :return: Standard TensorFlow feed dictionary which must be used at
            at initialization.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, session=None, force=False):
        """
        This method initializes all TensorFlow tensors listed by `initializables`
        property with the aid of feed dictionary presented by `initializable_feeds`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def build(self):
        """
        Public method for building tensors defined by ICompilable object at default
        TensorFlow graph. Wrapper for internal `_build` method.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compile(self, session=None):
        """
        Two-phase method. At first it builds tensors and then initializes them at
        for a specific session session.

        :param session: TensorFlow session.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_built(self, graph):
        """
        Checks if tensors belonging to this ICompilable object were built for
        the `graph` argument.

        :param graph: TensorFlow graph.
        :return: `Build` status.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        """
        Clears out tensors from ICompilable object and removes all ties with them.
        """
        raise NotImplementedError()
