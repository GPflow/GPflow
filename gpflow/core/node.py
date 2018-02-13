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
from .errors import GPflowError
from .compilable import Build
from .compilable import ICompilable
from .parentable import Parentable

class Node(Parentable, ICompilable):
    """
    The Node class merges two simple conceptions: _parentable structure_ with rich naming
    feature and _compilable interface_ which gives enough flexibility to build and
    run TensorFlow operations only forcing user to override only necessary building blocks.
    """

    def __init__(self, name=None):
        super(Node, self).__init__(name=name)

    @abc.abstractmethod
    def anchor(self, session):
        """
        The `anchor` method is intended to bind TensorFlow session values with
        pythonic values saved by node's objects.
        """
        raise NotImplementedError('Public method anchor must be implemented by successor.')

    def compile(self, session=None):
        """
        Compile is two phase operation: at first it calls `build` method and then
        intializes the node for passed session. The policy around `session` is defined
        inside the `initialize` method.

        :param session: TensorFlow session used for initializing. If the node is built the
            session's graph value must be equal to the node tensor's graph.

        :raises: GPflowError exception if session's graph is different from the graph
            used by node tensors.
        """
        session = self.enquire_session(session)
        if self.is_built_coherence(session.graph) is Build.NO:
            with session.graph.as_default():
                self.build()
        self.initialize(session, force=True)

    def initialize(self, session=None, force=False):
        """
        Initializes TensorFlow variables, which are returned by `initializables` property and
        uses feed dictionary returned by `initializable_feeds` property defined at ICompilable
        interface and implemented by descendants.

        :param session: TensorFlow session used for initializing. In case when session is None,
            default TensorFlow session will be checked first, if session is still None, then
            default GPflowFlow session will used, but there is *no garuantee* that GPflow
            session's graph is compliant with node's tensors graph.
        :param force: inidicates either the initialized TensorFlow variables must be
            re-initialized or not.

        :raises: GPflowError exception if session's graph is different from the graph
            used by node tensors.
        """
        session = self.enquire_session(session)
        initializables = self.initializables
        if initializables:
            misc.initialize_variables(
                variables=initializables,
                session=session,
                force=force,
                feed_dict=self.initializable_feeds)

    def clear(self):
        """
        Calls `_clear` abstract method which must be implemented by descendants.

        :raises: GPflowError exception when parent of the node is built.
        """
        parent = self.parent
        if parent is not self and parent.is_built_coherence(self.graph) is Build.YES:
            raise GPflowError('Clear method cannot be started. Upper nodes are built.')
        self._clear()

    def enquire_graph(self, graph=None):
        """
        Verifies and returns relevant TensorFlow graph. If non-None graph were passed,
        the same graph is returned. Otherwise, nodes's graph is exposed and it is
        undefined the default TensorFlow graph is used.

        :param graph: TensorFlow graph or None. Default is None.
        :return: TensorFlow graph.
        """
        if graph is None:
            graph = self.root.graph if self.graph is None else self.graph
            if graph is None:
                graph = tf.get_default_graph()
        return graph

    def enquire_session(self, session=None):
        """
        Verifies and returns relevant TensorFlow session. If non-None session
        were passed, session is checked for graph compliance and returned back.
        Otherwise, default TensorFlow session is returned. When TensorFlow default
        session is not set up, GPflow session's manager creates or uses existing
        one for returning.

        :param session: TensorFlow session or None. Default value is None.
        :return: TensorFlow session.
        :raises GPflowError: Session's graph is not compilable with node's graph.
        """
        if session is None:
            session = tf.get_default_session()
            if session is None:
                session = session_manager.get_default_session()
        self.is_built_coherence(session.graph)
        return session

    def is_built_coherence(self, graph=None):
        """
        Checks that node was build using input `graph`.

        :return: `Build` status.
        :raises GPflowError: Valid passed TensorFlow graph is different from
            used graph in node.
        """
        graph = self.enquire_graph(graph=graph)
        is_built = self.is_built(graph)
        if is_built is Build.NOT_COMPATIBLE_GRAPH:
            raise GPflowError('Tensor "{}" uses different graph.'.format(self.pathname))
        return is_built

    def build(self):
        """
        Implementation for ICompilable interface `build` method. Builds tensors within
        TensorFlow name scope using parentable node's name. Hidden name is used when 
        no parent exists for current node.

        :raises GPflowError: Node's parts were built with different graph and differ from
            default TensorFlow graph.
        """
        if self.is_built_coherence() is Build.NO:
            with tf.name_scope(self.tf_name_scope):
                self._build()
    
    @property
    def tf_name_scope(self):
        """
        Auxilary method for composing gpflow's tree name scopes. The Parentable pathname
        can be considered as a set of name scopes. This method grabs `pathname` and
        returns only name of the node in that path.
        Leading node name is always replaced with two parts: the name and the index
        for uniquiness in TensorFlow.
        """
        if self.parent is self:
            leader_name = self.name
            leader_index = self.index
            if leader_index is None:
                return leader_name
            return "{name}-{index}".format(name=leader_name, index=leader_index)
        return self.pathname.rsplit('/')[-1]

    @property
    def tf_pathname(self):
        """
        Method used for defining full path name for particular tensor at build time.
        For example, `tf.get_variable` creates variable w/o taking into account
        name scopes and `tf_pathname` consists of all parts of scope names
        which were used up to that point - `tf.get_variable` call.
        """
        if self.parent is self:
            return self.tf_name_scope
        tail = self.pathname.split('/', 1)[-1]
        leader = self.root.tf_name_scope
        return "{leader_name}/{tail_name}".format(leader_name=leader, tail_name=tail)

    @abc.abstractmethod
    def _clear(self):
        """
        Internal clear function. This method must be overridden by descendants with ICompilable
        compabilities instead of public `clear`.
        """
        raise NotImplementedError('Private method clear must be implemented by successor.')

    @abc.abstractmethod
    def _build(self):
        """
        Internal build function. This method must be overridden by descendants with ICompilable
        compabilities instead of public `build`.
        """
        raise NotImplementedError('Private method `build` must be implemented by successor.')