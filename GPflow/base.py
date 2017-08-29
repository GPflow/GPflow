import abc
from .misc import tensor_name, get_tensor_by_name

class IGraphOwner:
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def graph(self):
        pass

    @abc.abstractproperty
    def session(self):
        pass

class ICompilable:
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def is_compiled(self):
        pass

    @abc.abstractmethod
    def compile(self, graph=None):
        pass

    @abc.abstractmethod
    def destroy(self):
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
        raise NotImplementedError


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


class Parentable:
    """
    A very simple class for objects in a tree, where each node contains a
    reference to '_parent'.

    This class can figure out its own name (by seeing what it's called by the
    _parent's __dict__) and also recurse up to the root.
    """

    def __init__(self, name=None):
        self._parent = None
        self._name = self.__class__.__name__ if name is None else name

    @property
    def root(self):
        """A reference to the top of the tree, usually a Model instance"""
        if self._parent is None:
            return self
        return self._parent.root

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

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_parent')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._parent = None
