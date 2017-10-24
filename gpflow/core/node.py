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
import tensorflow as tf

from .. import session_manager
from ..core.base import Build
from ..core.base import GPflowError
from ..core.base import ICompilable
from ..core.parentable import Parentable


class Node(Parentable, ICompilable): # pylint: disable=W0223

    def __init__(self, name=None):
        super(Node, self).__init__(name=name)
        self._session = None

    @abc.abstractmethod
    def anchor(self):
        raise NotImplementedError('Public method anchor must be implemented by successor.')

    @abc.abstractmethod
    def _clear(self):
        raise NotImplementedError('Private method clear must be implemented by successor.')
    @property
    def session(self):
        if self._session is not None:
            return self._session
        root = self.root
        if root is self:
            return self._session
        return root.session

    def compile(self, session=None, keep_session=True):
        if self.parent is not self:
            raise GPflowError('Only root can initiate compilation.')
        session = self._setup_compile_session(session)
        if self.is_built(session.graph) is Build.NO:
            with session.graph.as_default():
                self._build_with_name_scope()
        self.initialize(session)
        if keep_session:
            self._session = session

    def clear(self):
        if self.root is not self:
            raise GPflowError('Only root can initiate clear process.')
        self._session = None
        self._clear()

    def enquire_graph(self, graph=None):
        if graph is None:
            graph = self.root.graph if self.graph is None else self.graph
            if graph is None:
                graph = tf.get_default_graph()
        return graph

    def enquire_session(self, session=None, allow_none=False):
        if session is None and self.session is None:
            if allow_none:
                return None
            raise ValueError('Session is not specified.')
        session = self.session if session is None else session
        if self.is_built_coherence(session.graph) is Build.NO:
            raise GPflowError('Not compiled.')
        return session

    def set_session(self, session):
        if not isinstance(session, tf.Session):
            raise ValueError('Argument is not session type.')
        if session is None:
            raise ValueError('Session is None.')
        if self.root is not self:
            raise ValueError('Session cannot be changed for non-root compilable node.')
        if self.is_built_coherence(session.graph) is Build.YES:
            self.initialize(session)
        self._session = session

    def set_parent(self, parent=None):
        if parent is self:
            raise ValueError('Self references are prohibited.')
        if parent and not isinstance(parent, Parentable):
            raise ValueError('Parent object must implement parentable interface.')
        self._session = None
        self._parent = parent if parent is not None else None

    def is_built_coherence(self, graph=None):
        graph = self.enquire_graph(graph=graph)
        if graph and self.session and self.session.graph is not graph:
            raise GPflowError('Tensor uses different graph.')
        is_built = self.is_built(graph)
        if is_built is Build.NOT_COMPATIBLE_GRAPH:
            raise GPflowError('Tensor uses different graph.')
        return is_built

    def _setup_compile_session(self, session):
        if session is None:
            session = self.session
            if session is None:
                session = tf.get_default_session()
            if session is None:
                graph = self.enquire_graph()
                session = session_manager.get_session(graph=graph)
        self.is_built_coherence(session.graph)
        return session

    def _build_with_name_scope(self, name=None):
        name = self.hidden_name if name is None else name
        is_built = self.is_built(tf.get_default_graph())
        if is_built is Build.NOT_COMPATIBLE_GRAPH:
            raise GPflowError('Tensor uses different graph.')
        elif is_built is Build.NO:
            with tf.name_scope(name):
                self._build()
