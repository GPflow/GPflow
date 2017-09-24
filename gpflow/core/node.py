import abc
import tensorflow as tf

from gpflow import session_manager
from gpflow.core.base import Build
from gpflow.core.base import GPflowError
from gpflow.core.base import Parentable, ICompilable


class Node(Parentable, ICompilable): # pylint: disable=W0223

    def __init__(self, name=None):
        super(Node, self).__init__(name=name)
        self._session = None

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
        if self.is_built_coherence() is Build.YES and self.root is not self:
            raise GPflowError('Only root can initiate cleaning process.')
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
        if self.parent is not self and self.is_built_coherence() is Build.YES:
            raise GPflowError('Parent cannot be changed for compiled node.')
        if parent and not isinstance(parent, Parentable):
            raise GPflowError('Argument does not implement parentable interface.')
        self._session = None
        self._parent = parent if parent is not None else self

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
        name = self.name if name is None else name
        is_built = self.is_built(tf.get_default_graph())
        if is_built is Build.NOT_COMPATIBLE_GRAPH:
            raise GPflowError('Tensor uses different graph.')
        elif is_built is Build.NO:
            with tf.name_scope(name):
                self._build()

    @abc.abstractmethod
    def _clear(self):
        raise NotImplementedError('Private method clear must be implemented by successor.')
