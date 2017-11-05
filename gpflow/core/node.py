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

from .. import misc
from .. import session_manager
from ..core.base import Build
from ..core.base import GPflowError
from ..core.base import ICompilable
from ..core.parentable import Parentable

class Node(Parentable, ICompilable):

    def __init__(self, name=None):
        super(Node, self).__init__(name=name)

    @abc.abstractmethod
    def anchor(self, session):
        raise NotImplementedError('Public method anchor must be implemented by successor.')

    @abc.abstractmethod
    def _clear(self):
        raise NotImplementedError('Private method clear must be implemented by successor.')

    def compile(self, session=None):
        session = self.enquire_session(session)
        if self.is_built(session.graph) is Build.NO:
            with session.graph.as_default():
                self.build()
        self.initialize(session)

    def initialize(self, session=None, force=False):
        session = self.enquire_session(session)
        initializables = self.initializables
        if initializables:
            misc.initialize_variables(
                variables=initializables,
                session=session,
                force=force,
                feed_dict=self.initializable_feeds)

    def clear(self):
        parent = self.parent
        if parent is not self and parent.is_built_coherence(self.graph) is Build.YES:
            raise GPflowError('Clear method cannot be started. Upper nodes are built.')
        self._clear()

    def enquire_graph(self, graph=None):
        if graph is None:
            graph = self.root.graph if self.graph is None else self.graph
            if graph is None:
                graph = tf.get_default_graph()
        return graph

    def enquire_session(self, session=None):
        if session is None:
            session = tf.get_default_session()
            if session is None:
                session = session_manager.get_default_session()
        self.is_built_coherence(session.graph)
        return session

    def set_parent(self, parent=None):
        if parent is self:
            raise ValueError('Self references are prohibited.')
        if parent and not isinstance(parent, Parentable):
            raise ValueError('Parent object must implement parentable interface.')
        self._parent = parent if parent is not None else None

    def is_built_coherence(self, graph=None):
        graph = self.enquire_graph(graph=graph)
        is_built = self.is_built(graph)
        if is_built is Build.NOT_COMPATIBLE_GRAPH:
            raise GPflowError('Tensor "{}" uses different graph.'.format(self.full_name))
        return is_built

    def build(self):
        if self.is_built_coherence() is Build.NO:
            name = self.hidden_name if self.parent is self else self.name
            with tf.name_scope(name):
                self._build()
