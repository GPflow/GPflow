import numpy as np
import tensorflow as tf
import transforms
from contextlib import contextmanager

recompile_keys = ['prior', 'transform', 'fixed'] # when one of these attributes is set, notify a recompilation

class Parentable(object):
    """
    A very simple class for objects in a tree, where each node contains a
    reference to '_parent'. 

    This class can figure out its own name (by seeing what it's called by the
    _parent's __dict__) and also recurse up to the highest_parent.
    """
    _parent = None
    @property
    def highest_parent(self):
        if self._parent is None:
            return self
        else:
            return self._parent.highest_parent
    @property
    def name(self):
        """to get the name of this object, have a look at what our _parent has called us"""
        if self._parent is None:
            return 'unnamed'
        if isinstance(self._parent, ParamList):
            return 'item%i'%self._parent._list.index(self)
        matches = [key for key, value in self._parent.__dict__.items() if value is self]
        if len(matches) == 0:
            raise ValueError, "mis-specified parent. This param's _parent does not contain a reference to it."
        if len(matches) > 1:
            raise ValueError, "This param appears to be doubly referenced by a parent"
        return matches [0]



class Param(Parentable):
    """
    An object to represent parameters. 


    Getting and setting values
    --
    The current value of the parameter is stored in self._array as a
    numpy.ndarray.  Changing the value of the Param is as simple as assignment
    (once the Param is part of a model). Example:

    >>> m = GPflow.model.Model()
    >>> m.p = GPflow.param.Param(1.0)
    >>> print(m)
    model.p transform:(none) prior:None
    [ 1.]
    >>> m.p = 3.2
    >>> print(m)
    model.p transform:(none) prior:None
    [ 3.2]

    Unconstrained optimization
    --
    The parameter can be transformed to a 'free state' where it
    can be optimized. The methods

    >>> self.get_free_state
    >>> self.set_state
    
    transforms between self._array and the free state. 

    To apply a transform to the Param, simply set the transform atribute with a GPflow.transforms object
    >>> m = GPflow.model.Model()
    >>> m.p = GPflow.param.Param(1.0)
    >>> print(m)
    model.p transform:(none) prior:None
    [ 1.]
    >>> m.p.transform = GPflow.transforms.Exp()
    >>> print(m)
    model.p transform:+ve prior:None
    [ 1.]


    Fixes
    --
    There is a self._fixed flag, in which case the parameter does not get
    optimized. To enable this, during make_tf_array, the fixed values of
    the parameter are returned. Fixes and transforms can be used together, in
    the sense that fixes tkae priority over transforms, so unfixing a parameter
    is as simple as setting the flag. Example:

    >>> p = Param(1.0, transform=GPflow.transforms.positive)
    >>> m = GPflow.model.Model()
    >>> m.p = p # the model has a sinlge parameter, constrained to be +ve
    >>> m.p.fixed = True # the model now has no free parameters
    >>> m.p.fixed = False # the model has a sinlge parameter, constrained to be +ve


    Compiling into tensorflow
    --
    The method 
        
    >>> self.make_tf_array

    constructs a tensorflow representation of the parameter, from a tensorflow vector representing the free state. 

    The `self.prior` object is used to place priors on prameters, and the
    `self.transform` object is used to enable unconstrained optimization and
    mcmc.



    """
    def __init__(self, array, transform=transforms.Identity()):
        Parentable.__init__(self)
        self._array = np.asarray(np.atleast_1d(array), dtype=np.float64)
        self.transform = transform
        self.prior = None
        self.fixed = False

    def make_tf_array(self, free_array):
        """
        free_array is a tensorflow vector which will be the optimisation target,
        i.e. it will be free to take any value.

        Here we take that array, and transform and reshape it so that it can be
        used to represent this parameter

        Then we return the number of elements that we've used to construct the
        array, so that it can be sliced fo rthe next Param.
        """

        #TODO what about constraints that change the size ??

        if self.fixed:
            self._tf_array = self._array.copy()
            return 0
        x_free = free_array[:self.size]
        mapped_array = self.transform.tf_forward(x_free)
        self._tf_array = tf.reshape(mapped_array, self.shape)
        self._log_jacobian = self.transform.tf_log_jacobian(x_free)
        return self.size

    def get_free_state(self):
        """
        Take the current state of this variable, as stored in self._array, and transform it to the 'free' state.

        This is a numpy method.
        """
        if self.fixed: return np.empty((0,))
        return self.transform.backward(self._array.flatten())

    def set_state(self, x):
        """
        Given a vector x representing the 'free' state of this param, transform
        it 'forwards' and store the result in self._array. 

        This is a numpy method.
        """
        if self.fixed: return 0
        new_array = self.transform.forward(x[:self.size]).reshape(self.shape)
        assert new_array.shape == self.shape
        self._array[...] = new_array
        return self.size

    def build_prior(self):
        """
        Build a tensorflow representation of the prior density. The log jacobian is included. 
        """
        if self.prior is None:
            return 0
        elif self._tf_array is None:
            raise ValueError, "tensorflow array has not been initialized"
        else:
            return self.prior.logp(self._tf_array) + self._log_jacobian

    def __setattr__(self, key, value):
        """
        When some attirbutes are set, we need to recompile the tf model before evaluation.
        """
        object.__setattr__(self, key, value)
        if key in recompile_keys:
            self.highest_parent._needs_recompile = True

    def __str__(self, prepend=''):
        return prepend + \
                '\033[1m' + self.name + '\033[0m' + \
                ' transform:' + str(self.transform) + \
                ' prior:' + str(self.prior) + \
                (' [FIXED]' if self.fixed else '') + \
                '\n' + str(self._array)

    @property
    def size(self):
        return self._array.size

    @property
    def shape(self):
        return self._array.shape

    def _html_table_rows(self, name_prefix=''):
        """
        Construct a row of an html table, to be used in the jupyter notebook.
        """
        html = "<tr>"
        html += "<td>{0}</td>".format(name_prefix + self.name)
        html += "<td>{0}</td>".format(str(self._array).replace('\n', '</br>'))
        html += "<td>{0}</td>".format(str(self.prior))
        html += "<td>{0}</td>".format('[FIXED]' if self.fixed else str(self.transform))
        html += "</tr>"
        return html


class Parameterized(Parentable):
    """
    An object to contain parameters. 

    This object is designed to be part of a tree, with Param objects at the
    leaves. We can then recurse down the tree to find all the parameters
    (leaves), or recurse up the tree (using highest_parent) from the leaves to
    the root. 

    A useful application of such a recursion is 'tf_mode', where the
    parameters appear as their _tf_array variables. This allows us to build
    models on those parameters. During _tf_mode, the __getattribute__
    method is overwritten to return tf arrays in place of parameters.

    Another recurseive function is build_prior wich sums the log-prior from all
    of the tree's parameters (whilst in tf_mode!).
    """
    def __init__(self):
        Parentable.__init__(self)
        self._tf_mode = False

    def __getattribute__(self, key):
        """
        Here, we overwrite the getattribute method. 

        If tf mode is off, this does nothing.

        If tf mode is on, all child parameters will appear as their tf
        representations.
        """
        o =  object.__getattribute__(self, key)
        if isinstance(o, Param) and object.__getattribute__(self, '_tf_mode'):
            return o._tf_array
        return o

    def __setattr__(self, key, value):
        """
        When a value is assigned to a Param, put that value in the
        Param's array (rather than just overwriting that Param with the
        new value). i.e. this

        >>> p = Parameterized()
        >>> p.p = Param(1.0)
        >>> p.p = 2.0

        should be equivalent to this

        >>> p = Parameterized()
        >>> p.p = Param(1.0)
        >>> p.p._array[...] = 2.0

        Additionally, when Param or Parameterized objects are added, let them
        know that this node is the _parent
        """

        #set the _array value of child nodes instead of standard assignment.
        if key in self.__dict__.keys():
            p = getattr(self, key)
            if isinstance(p, Param):
                p._array[...] = value
                return # don't call object.setattr or set the _parent value

        #use the standard setattr
        object.__setattr__(self, key, value)

        #make sure a new child node knows this is the _parent:
        if isinstance(value, (Param, Parameterized)) and (key is not '_parent'):
            value._parent = self


    def make_tf_array(self, X):
        """
        X is a tf. placeholder. It gets passed to all the children of
        this class (that are Parameterized or Param objects), which then
        construct their tf_array variables from consecutive sections.
        """
        count = 0
        for p in self.sorted_params:
            count += p.make_tf_array(X[count:])
        return count

    @property
    def sorted_params(self):
        """
        Return a list of all the child parameters, sorted by id. This makes
        sure they're always in the same order. 
        """
        params = [child for key, child in self.__dict__.items() if isinstance(child, (Param, Parameterized)) and key is not '_parent']
        return sorted(params, key=id)

    def get_free_state(self):
        """
        recurse get_free_state on all child parameters, and hstack them.
        """
        return np.hstack([p.get_free_state() for p in self.sorted_params] + [np.empty(0)]) # additional empty array allows hstacking of empty list
    
    def set_state(self, x):
        """
        Set the values of all the parameters by recursion
        """
        count = 0
        for p in self.sorted_params:
            count += p.set_state(x[count:])
        return count

    @contextmanager
    def tf_mode(self):
        """
        A context for building models. Correct usage is

        with m.tf_mode:
            #do tf stuff, lik
            m.build_likelihood()
            m.build_prior()


        with this context engaged, any Param objects which are children of this
        class will appear as their tf-variables. Example

        >>> m = Parameterized()
        >>> m.foo = Param(1.0)
        >>> m.make_tf_array(tt.dvector())
        >>> print m.foo
        foo
        [ 1.]
        >>> with m.tf_mode():
        >>>     print m.foo
        Reshape{1}.0

        The idea is that in tf_mode, we can easily get refrences to the
        tf representation of parameters in order to construct tf
        objective functions.
        """
        self._begin_tf_mode()
        yield
        self._end_tf_mode()

    def _begin_tf_mode(self):
        [child._begin_tf_mode() for key, child in self.__dict__.items() if isinstance(child, Parameterized) and key is not '_parent']
        self._tf_mode = True

    def _end_tf_mode(self):
        [child._end_tf_mode() for key, child in self.__dict__.items() if isinstance(child, Parameterized) and key is not '_parent']
        self._tf_mode = False

    def build_prior(self):
        """
        Build a tf expression for the prior by summing all child-node priors.
        """
        return sum([p.build_prior() for p in self.sorted_params])

    def __str__(self, prepend=''):
        prepend += self.name + '.'
        return '\n'.join([p.__str__(prepend) for p in self.sorted_params])

    def _html_table_rows(self, name_prefix=''): # pragma: no cover
        """
        Get the rows of the html table for this object
        """
        name_prefix +=  self.name + '.'
        return ''.join([p._html_table_rows(name_prefix) for p in self.sorted_params])

    def _repr_html_(self):
        """
        Build a small html table for display in the jupyter notebook.
        """
        html = ["<table id='parms' width=100%>"]

        #build the header
        header = "<tr>"
        header += "<td>Name</td>"
        header += "<td>values</td>"
        header += "<td>prior</td>"
        header += "<td>constriant</td>"
        header += "</tr>"
        html.append(header)

        html.append(self._html_table_rows())

        html.append("</table>")
        return ''.join(html) 


class ParamList(Parameterized):
    """
    A list of parameters.
    """
    def __init__(self, list_of_params=[]):
        Parameterized.__init__(self)
        for item in list_of_params:
            assert isinstance(item, (Param, Parameterized))
            item._parent = self
        self._list = list_of_params

    @property
    def sorted_params(self):
        return self._list

    def __getitem__(self, key):
        """
        If tf mode is off, this simply returns the corresponding Param . 

        If tf mode is on, all items will appear as their tf
        representations.
        """
        o = self.sorted_params[key]
        if isinstance(o, Param) and self._tf_mode:
            return o._tf_array
        return o

    def append(self, item):
        assert isinstance(item, (Param, Parameterized)), "this object is for containing parameters"
        item._parent = self
        self.sorted_params.append(item)

    def __setitem__(self, key, value):
        """
        It's not possible to assign to things in the list, but it is possbile
        to set their values by assignment.
        """
        self.sorted_params[key]._array[...] = value
