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
            __execute_autobuild__ = AutoBuildStatus.BUILD if autobuild else AutoBuildStatus.IGNORE
            tag = AutoBuild._tag
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
