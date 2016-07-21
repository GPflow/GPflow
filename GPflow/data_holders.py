import tensorflow as tf
import numpy as np
import abc
from .tree_structure import Parentable


class DataHolder(Parentable):
    """
    Abstract base class for data holders.
    Cannot be instantiated directly
    Instead inherit from this class and implement virtual functions.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        Parentable.__init__(self)

    @abc.abstractmethod
    def get_feed_dict(self):
        """
        Return a dictionary matching up any fixed-placeholders to their values
        """
        raise NotImplementedError

    def __getstate__(self):
        d = Parentable.__getstate__(self)
        d.pop('_tf_array')
        return d

    def __setstate__(self, d):
        Parentable.__setstate__(self, d)

    def __str__(self, prepend='Data:'):
        return prepend + \
            '\033[1m' + self.name + '\033[0m' + \
            '\n' + str(self.value)


class TensorData(DataHolder):
    """
    A class to allow users to incorporate their own tensors into GPflow objects.
    """
    def __init__(self, dataTensor):
        """
        This object is instantiated by passing in a tensorflow array. GPflow
        will then incorporate this array into the model at compile time (e.g.
        during a call to build_likelihood)
        """
        DataHolder.__init__(self)
        self._tf_array = dataTensor

    def __setstate__(self, d):
        """
        It's not clear how to unpickle a model with user-controlled arrays, so
        raise an exception here.
        """
        DataHolder.__setstate__(self, d)
        raise NotImplementedError

    def get_feed_dict(self):
        # All done using a TensorFlow tensor so
        # there is no feed_dict associated with TensorData
        return {}

    def __str__(self, prepend='Data:'):
        return prepend + \
            '\033[1m' + self.name + '\033[0m' + \
            '\n' + '(user defined tensor)'


class DictData(DataHolder):
    def __init__(self, array, on_shape_change='raise'):
        """
        array is a numpy array of data.
        on_shape_change is one of ('raise', 'pass', 'recompile'), and
        determines the behaviour when the data is set to a new value with a
        different shape
        """
        DataHolder.__init__(self)
        self._array = array
        self._tf_array = tf.placeholder(dtype=self._array.dtype,
                                        shape=[None]*self._array.ndim,
                                        name=self.name)
        assert on_shape_change in ['raise', 'pass', 'recompile']
        self.on_shape_change = on_shape_change

    def set_data(self, array):
        """
        Setting a data into self._array before any TensorFlow execution.
        If the shape of the data changes, then either:
         - raise an exception
         - raise the recompilation flag.
         - do nothing
        according to the option in self.on_shape_change.
        """
        if self.shape == array.shape:
            self._array[...] = array  # just accept the new values
        else:
            if self.on_shape_change == 'raise':
                raise ValueError("The shape of this data must not change. \
                                  (perhaps make the model again from scratch?)")
            elif self.on_shape_change == 'recompile':
                self._array = array.copy()
                self.highest_parent._needs_recompile = True
                if hasattr(self.highest_parent, '_kill_autoflow'):
                    self.highest_parent._kill_autoflow()
            elif self.on_shape_change == 'pass':
                self._array = array.copy()
            else:
                raise ValueError('invalid option')  # pragma: no-cover

    @property
    def value(self):
        return self._array.copy()

    @property
    def size(self):
        return self._array.size

    @property
    def shape(self):
        return self._array.shape

    def __setstate__(self, d):
        DataHolder.__setstate__(self, d)
        tf_array = tf.placeholder(dtype=self._array.dtype,
                                  shape=[None]*self._array.ndim,
                                  name=self.name)
        self._tf_array = tf_array

    def get_feed_dict(self):
        return {self._tf_array: self._array}


class MinibatchData(DictData):
    """
    A special DataHolder class which feeds a minibatch to tensorflow via
    get_feed_dict().
    """
    def __init__(self, array, minibatch_size, rng=None):
        """
        array is a numpy array of data.
        minibatch_size (int) is the size of the minibatch
        rng is an instance of np.random.RandomState(), defaults to seed 0.
        """
        DictData.__init__(self, array, on_shape_change='pass')
        self.minibatch_size = minibatch_size
        self.rng = rng or np.random.RandomState(0)

    def generate_index(self):
        if float(self.minibatch_size) / float(self._array.shape[0]) > 0.5:
            return self.rng.permutation(self._array.shape[0])[:self.minibatch_size]
        else:
            # This is much faster than above, and for N >> minibatch,
            # it doesn't make much difference. This actually
            # becomes the limit when N is around 10**6, which isn't
            # uncommon when using SVI.
            return self.rng.randint(self._array.shape[0], size=self.minibatch_size)

    def get_feed_dict(self):
        return {self._tf_array: self._array[self.generate_index()]}
