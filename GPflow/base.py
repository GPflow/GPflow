import abc
import enum

from contextlib import contextmanager
import tensorflow as tf

from . import session_manager
from .misc import tensor_name
from .misc import GPflowError


class ICompilable:
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def graph(self):
        pass

    @abc.abstractmethod
    def compile(self, session=None):
        pass

    @abc.abstractmethod
    def initialize(self, session=None):
        pass

    @abc.abstractmethod
    def is_built(self, graph=None):
        pass

    @abc.abstractmethod
    def reset(self, graph=None):
        pass


class IPrior:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def logp(self, x):
        """
        The log density of the prior as x

        All priors (for the moment) are univariate, so if x is a vector or an
        array, this is the sum of the log densities.
        """
        pass

    @abc.abstractmethod
    def sample(self, shape=(1,)):
        """
        A sample utility function for the prior.
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        A short string to describe the prior at print time
        """
        pass


class ITransform:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, x):
        """
        Map from the free-space to the variable space, using numpy
        """
        pass

    @abc.abstractmethod
    def backward(self, y):
        """
        Map from the variable-space to the free space, using numpy
        """
        pass

    @abc.abstractmethod
    def tf_forward(self, x):
        """
        Map from the free-space to the variable space, using tensorflow
        """
        pass

    @abc.abstractmethod
    def tf_log_jacobian(self, x):
        """
        Return the log Jacobian of the tf_forward mapping.

        Note that we *could* do this using a tf manipulation of
        self.tf_forward, but tensorflow may have difficulty: it doesn't have a
        Jacobian at time of writing. We do this in the tests to make sure the
        implementation is correct.
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """
        A short string describing the nature of the constraint
        """
        raise NotImplementedError


class Build(enum.Enum):
    YES = 1
    NO = 0 # pylint: disable=C0103
    NOT_COMPATIBLE_GRAPH = None


class Parentable:
    """
    A very simple class for objects in a tree, where each node contains a
    reference to '_parent'.

    This class can figure out its own name (by seeing what it's called by the
    _parent's __dict__) and also recurse up to the root.
    """

    def __init__(self, name=None):
        self._parent = None
        self._name = self._define_name(name)

    @property
    def root(self):
        """A reference to the top of the tree, usually a Model instance"""
        if self._parent is None:
            return self
        return self._parent.root

    @property
    def parent(self):
        if self._parent is None:
            return self
        return self._parent

    @property
    def name(self):
        return self._name

    @property
    def full_name(self):
        """
        This is a unique identifier for a param object within a structure, made
        by concatenating the names through the tree.
        """
        if self._parent is None:
            return self.name
        return tensor_name(self._parent.full_name, self.name)

    def set_name(self, name=None):
        self._name = self._define_name(name)

    def set_parent(self, parent=None):
        self._parent = parent

    def _define_name(self, name):
        return self.__class__.__name__ if name is None else name

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_parent')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._parent = None


class CompilableNode(Parentable, ICompilable): # pylint: disable=W0223

    def __init__(self, name=None):
        super(CompilableNode, self).__init__(name=name)
        self._initiator = None
        self._session = None

    @property
    def session(self):
        if self._session is not None:
            return self._session
        if self.root is self:
            return self._session
        return self.root.session


    def compile(self, session=None, keep_session=True):
        session = self.setup_compile_session(session, keep_session)
        is_built = self.is_built_coherence(graph=session.graph)
        with session.graph.as_default():
            self._build_with_name_scope()
        self.initialize(session)

    def verified_graph(self, graph=None):
        if graph is None:
            graph = self.root.graph if self.graph is None else self.graph
            if graph is None:
                graph = tf.get_default_graph()
        return graph

    def verified_compile_session(self, session):
        if session is None:
            session = self.session
            if session is None:
                session = tf.get_default_session()
            if session is None:
                graph = self.verified_graph()
                session = session_manager.get_session(graph=graph)
        _ = self.is_built_coherence(session.graph)
        return session

    def verified_custom_session(self, session):
        if session is None and self.session is None:
            raise ValueError('Session is not specified.')
        session = session if session is not None else self.session
        if self.is_built_coherence(session.graph) is Build.NO:
            raise GPflowError('Not compiled.')
        return session

    def set_session(self, session):
        if session is None:
            raise ValueError('Session is None.')
        if self.root is not self:
            raise ValueError('Session cannot be changed for non-root compilable object.')
        if self.is_built_coherence(session.graph) is Build.YES:
            self.initialize(session)
        self._session = session

    def set_parent(self, parent=None):
        if parent is None:
            self._session = self.session
        else:
            if parent.graph and parent.is_built_coherence(self.graph) is Build.YES:
                raise GPflowError('Parent is assembled.')
            # TODO(awav): do we need implicitly pass session from child to parent
            if parent.session is None and self.session:
                parent.set_session(self.session)
        self._parent = parent

    def _build(self):
        raise NotImplementedError()

    def _build_with_name_scope(self, name=None):
        name = self.name if name is None else name
        with tf.name_scope(name):
            self._build()

    def is_built_coherence(self, graph=None):
        if self.session and graph and self.session.graph is not graph:
            raise GPflowError('Tensor uses different graph.')
        is_built = self.is_built(graph)
        if is_built is Build.NOT_COMPATIBLE_GRAPH:
            raise GPflowError('Tensor uses different graph.')
        return is_built

    @contextmanager
    def compilation_context(self, session):
        if session is None:
            raise ValueError('Passed session is None.')
        with self._context_as_default(session.graph):
            yield session
            self._exit_compilation(session)

    @contextmanager
    def _context_as_default(self, graph):
        if graph is None:
            raise ValueError('Passed graph is None.')
        if graph is tf.get_default_graph():
            with tf.name_scope(self.name):
                yield
        else:
            with graph.as_default(), tf.name_scope(self.name):
                yield

    def _exit_compilation(self):
        if self.parent is self:
            self.initialize(session)
