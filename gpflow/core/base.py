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

import tensorflow as tf


class GPflowError(Exception):
    pass


class Build(enum.Enum):
    YES = 1
    NO = 0  # pylint: disable=C0103
    NOT_COMPATIBLE_GRAPH = None


class _AutoBuildStatus(enum.Enum):
    # TODO(@awav): Introducing global variable is not best idea for managing compilation.
    # Inspect approach works well, except that I have a concern that it can be slow when
    # nesting is too deep and it doesn't work well with context managers.

    __autobuild_enabled_global__ = True

    BUILD = 1
    IGNORE = 2
    FOLLOW = 3


class AutoBuild(abc.ABCMeta):
    _autobuild_arg = 'autobuild'
    _tag = '__execute_autobuild__'

    def __new__(mcs, name, bases, namespace, **kwargs):
        new_cls = super(AutoBuild, mcs).__new__(mcs, name, bases, namespace, **kwargs)
        origin_init = new_cls.__init__
        def __init__(self, *args, **kwargs):
            autobuild = kwargs.pop(AutoBuild._autobuild_arg, True)
            __execute_autobuild__ = _AutoBuildStatus.BUILD if autobuild else _AutoBuildStatus.IGNORE
            tag = AutoBuild._tag
            frame = inspect.currentframe().f_back
            while autobuild and frame:
                if isinstance(frame.f_locals.get(tag, None), _AutoBuildStatus):
                    __execute_autobuild__ = _AutoBuildStatus.FOLLOW
                    break
                frame = frame.f_back
            origin_init(self, *args, **kwargs)
            autobuild_on = __execute_autobuild__ == _AutoBuildStatus.BUILD
            global_autobuild_on = _AutoBuildStatus.__autobuild_enabled_global__
            if autobuild_on and global_autobuild_on:
                self.build()
                self.initialize(force=True)
        setattr(new_cls, '__init__', __init__)
        return new_cls


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


class IPrior(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def logp(self, x):
        """
        The log density of the prior as x

        All priors (for the moment) are univariate, so if x is a vector or an
        array, this is the sum of the log densities.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self, shape=(1,)):
        """
        A sample utility function for the prior.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        """
        A short string to describe the prior at print time
        """
        raise NotImplementedError()


class ITransform(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def forward(self, x):
        """
        Map from the free-space to the variable space, using numpy
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backward(self, y):
        """
        Map from the variable-space to the free space, using numpy
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_tensor(self, x):
        """
        Map from the free-space to the variable space, using tensorflow
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_jacobian_tensor(self, x):
        """
        Return the log Jacobian of the forward_tensor mapping.

        Note that we *could* do this using a tf manipulation of
        self.forward_tensor, but tensorflow may have difficulty: it doesn't have a
        Jacobian at time of writing. We do this in the tests to make sure the
        implementation is correct.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        """
        A short string describing the nature of the constraint
        """
        raise NotImplementedError
