import numpy as np
import pandas as pd
import tensorflow as tf
from . import transforms
from contextlib import contextmanager
from functools import wraps

# when one of these attributes is set, notify a recompilation
recompile_keys = ['prior', 'transform', 'fixed']


class Parentable(object):
    """
    A very simple class for objects in a tree, where each node contains a
    reference to '_parent'.

    This class can figure out its own name (by seeing what it's called by the
    _parent's __dict__) and also recurse up to the highest_parent.
    """

    def __init__(self):
        self._parent = None

    @property
    def highest_parent(self):
        if self._parent is None:
            return self
        else:
            return self._parent.highest_parent

    @property
    def name(self):
        """to get the name of this object, have a look at
        what our _parent has called us"""
        if self._parent is None:
            return 'unnamed'
        if isinstance(self._parent, ParamList):
            return 'item%i' % self._parent._list.index(self)
        matches = [key for key, value in self._parent.__dict__.items()
                   if value is self]
        if len(matches) == 0:
            raise ValueError("mis-specified parent. This Param's\
                             _parent does not contain a reference to it.")
        if len(matches) > 1:
            raise ValueError("This Param appears to be doubly\
                             referenced by a parent")
        return matches[0]

    @property
    def long_name(self):
        """
        This is a unique identifier for a param object within a structure, made
        by concatenating the names through the tree.
        """
        if self._parent is None:
            return self.name
        return self._parent.long_name + '.' + self.name

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('_parent')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._parent = None


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

    To retrieve the value of the parameter, we use the 'value' property:
    >>> m.p.value
    array([ 3.2])

    Unconstrained optimization
    --
    The parameter can be transformed to a 'free state' where it
    can be optimized. The methods

    >>> self.get_free_state
    >>> self.set_state

    transform between self.value and the free state.

    To apply a transform to the Param, simply set the transform attribute
    with a GPflow.transforms object
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
    There is a self.fixed flag, in which case the parameter does not get
    optimized. To enable this, during make_tf_array, a fixed parameter will be
    ignored, and a placeholder returned via get_feed_dict instead.

    Fixes and transforms can be used together, in the sense that fixes take
    priority over transforms, so unfixing a parameter is as simple as setting
    the flag. Example:

    >>> p = Param(1.0, transform=GPflow.transforms.positive)
    >>> m = GPflow.model.Model()
    >>> m.p = p # the model has a single parameter, constrained to be +ve
    >>> m.p.fixed = True # the model now has no free parameters
    >>> m.p.fixed = False # the model has a single parameter, constrained +ve

    Note that if self.fixed flag is assigned,  recompilation of the model is
    necessary.  Otherwise, the change in the fixed parameter values does not
    require recompilation.


    Compiling into tensorflow
    --
    The method

    >>> self.make_tf_array

    constructs a tensorflow representation of the parameter, from a tensorflow
    vector representing the free state. In this case, the parameter is
    represented as part of the 'free-state' vector associated with a model.
    However, if the parameters is fixed, then a placeholder is returned during
    a call to get_feed_dict, and the parameter is represented that way instead.

    Priors and transforms
    --
    The `self.prior` object is used to place priors on parameters, and the
    `self.transform` object is used to enable unconstrained optimization and
    MCMC.
    """

    def __init__(self, array, transform=transforms.Identity()):
        Parentable.__init__(self)
        self._array = np.asarray(np.atleast_1d(array), dtype=np.float64)
        self.transform = transform
        self._tf_array = None
        self._log_jacobian = None
        self.prior = None
        self.fixed = False

    @property
    def value(self):
        return self._array.copy()

    def get_parameter_dict(self, d):
        d[self.long_name] = self.value

    def set_parameter_dict(self, d):
        self._array[...] = d[self.long_name]

    def get_samples_df(self, samples):
        """
        Given a numpy array where each row is a valid free-state vector, return
        a pandas.DataFrame which contains the parameter name and associated samples
        in the correct form (e.g. with positive constraints applied).
        """
        if self.fixed:
            return pd.Series([self.value for _ in range(samples.shape[0])], name=self.long_name)
        start, _ = self.highest_parent.get_param_index(self)
        end = start + self.size
        samples = samples[:, start:end]
        samples = samples.reshape((samples.shape[0],) + self.shape)
        samples = self.transform.forward(samples)
        return pd.Series([v for v in samples], name=self.long_name)

    def make_tf_array(self, free_array):
        """
        free_array is a tensorflow vector which will be the optimisation
        target, i.e. it will be free to take any value.

        Here we take that array, and transform and reshape it so that it can be
        used to represent this parameter

        Then we return the number of elements that we've used to construct the
        array, so that it can be sliced for the next Param.
        """

        # TODO what about constraints that change the size ??

        if self.fixed:
            # fixed parameters are treated by tf.placeholder
            return 0
        x_free = free_array[:self.size]
        mapped_array = self.transform.tf_forward(x_free)
        self._tf_array = tf.reshape(mapped_array, self.shape)
        self._log_jacobian = self.transform.tf_log_jacobian(x_free)
        return self.size

    def get_free_state(self):
        """
        Take the current state of this variable, as stored in self.value, and
        transform it to the 'free' state.

        This is a numpy method.
        """
        if self.fixed:
            return np.empty((0,))
        return self.transform.backward(self.value.flatten())

    def get_feed_dict(self):
        """
        Return a dictionary matching up any fixed-placeholders to their values
        """
        d = {}
        if self.fixed:
            d[self._tf_array] = self.value
        return d

    def set_state(self, x):
        """
        Given a vector x representing the 'free' state of this Param, transform
        it 'forwards' and store the result in self._array. The values in
        self._array can be accessed using self.value

        This is a numpy method.
        """
        if self.fixed:
            return 0
        new_array = self.transform.forward(x[:self.size]).reshape(self.shape)
        assert new_array.shape == self.shape
        self._array[...] = new_array
        return self.size

    def build_prior(self):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if self.prior is None:
            return tf.constant(0.0, tf.float64)
        elif self._tf_array is None:  # pragma: no-cover
            raise ValueError("tensorflow array has not been initialized")
        else:
            return self.prior.logp(self._tf_array) + self._log_jacobian

    def __setattr__(self, key, value):
        """
        When some attributes are set, we need to recompile the tf model before
        evaluation.
        """
        object.__setattr__(self, key, value)
        if key in recompile_keys:
            self.highest_parent._needs_recompile = True

        # when setting the fixed attribute, make or remove a placeholder appropraitely
        if key == 'fixed':
            if value:
                self._tf_array = tf.placeholder(dtype=tf.float64,
                                                shape=self._array.shape,
                                                name=self.name)
            else:
                self._tf_array = None

    def __str__(self, prepend=''):
        return prepend + \
            '\033[1m' + self.name + '\033[0m' + \
            ' transform:' + str(self.transform) + \
            ' prior:' + str(self.prior) + \
            (' [FIXED]' if self.fixed else '') + \
            '\n' + str(self.value)

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
        html += "<td>{0}</td>".format('[FIXED]' if self.fixed
                                      else str(self.transform))
        html += "</tr>"
        return html

    def __getstate__(self):
        d = Parentable.__getstate__(self)
        d.pop('_tf_array')
        d.pop('_log_jacobian')
        return d

    def __setstate__(self, d):
        Parentable.__setstate__(self, d)
        self._tf_array = None
        self._log_jacobian = None
        # NB the parent property will be set by the parent object, apart from
        # for the top level, where it muct be None
        # the tf_array and _log jacobian will be replaced when the model is recompiled


class DataHolder(Parentable):
    """
    An object to represent data which needs to be passed to tensorflow for computation.

    This behaves in much the same way as a Param (above), but is always
    'fixed'. On a call to get_feed_dict, a placeholder-numpy pair is returned.

    Getting and setting values
    --
    To get at the values of the data, use the value property:

    >>> m = GPflow.model.Model()
    >>> m.x = GPflow.param.DataHolder(np.array([ 0., 1.]))
    >>> print(m.x.value)
    [[ 0.], [ 1.]]

    Changing the value of the data is as simple as assignment
    (once the data is part of a model):

    >>> m.x = np.array([ 0., 2.])
    >>> print(m.x.value)
    [[ 0.], [ 2.]]

    """
    def __init__(self, array, on_shape_change='raise'):
        """
        array is a numpy array of data.
        on_shape_change is one of ('raise', 'pass', 'recompile'), and
        determines the behaviour when the data is set to a new value with a
        different shape
        """
        Parentable.__init__(self)
        self._array = array
        self._tf_array = tf.placeholder(dtype=self._array.dtype,
                                        shape=[None]*self._array.ndim,
                                        name=self.name)
        assert on_shape_change in ['raise', 'pass', 'recompile']
        self.on_shape_change = on_shape_change

    def get_feed_dict(self):
        return {self._tf_array: self._array}

    def __getstate__(self):
        d = Parentable.__getstate__(self)
        d.pop('_tf_array')
        return d

    def __setstate__(self, d):
        Parentable.__setstate__(self, d)
        self._tf_array = tf.placeholder(dtype=self._array.dtype,
                                        shape=[None]*self._array.ndim,
                                        name=self.name)

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

    def __str__(self, prepend='Data:'):
        return prepend + \
            '\033[1m' + self.name + '\033[0m' + \
            '\n' + str(self.value)


class AutoFlow:
    """
    This decorator-class is designed for use on methods of the Parameterized class
    (below).

    The idea is that methods that compute relevant quantities (such as
    predictions) can define a tf graph which we automatically run when the
    (decorated) function is called.

    The syntax looks like:

    >>> class MyClass(Parameterized):
    >>>
    >>>   @AutoFlow((tf.float64), (tf.float64))
    >>>   def my_method(self, x, y):
    >>>       #compute something, returning a tf graph.
    >>>       return tf.foo(self.baz, x, y)

    >>> m = MyClass()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> result = my_method(x, y)

    Now the output of the method call is the _result_ of the computation,
    equivalent to

    >>> m = MyModel()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> x_tf = tf.placeholder(tf.float64)
    >>> y_tf = tf.placeholder(tf.float64)
    >>> with m.tf_mode():
    >>>     graph = tf.foo(m.baz, x_tf, y_tf)
    >>> result = m._session.run(graph,
                                feed_dict={x_tf:x,
                                           y_tf:y,
                                           m._free_vars:m.get_free_state()})

    Not only is the syntax cleaner, but multiple calls to the method will
    result in the graph being constructed only once.

    """

    def __init__(self, *tf_arg_tuples):
        # NB. TF arg_tuples is a list of tuples, each of which can be used to
        # construct a tf placeholder.
        self.tf_arg_tuples = tf_arg_tuples

    def __call__(self, tf_method):
        @wraps(tf_method)
        def runnable(instance, *np_args):
            storage_name = '_' + tf_method.__name__ + '_AF_storage'
            if hasattr(instance, storage_name):
                # the method has been compiled already, get things out of storage
                storage = getattr(instance, storage_name)
            else:
                # the method needs to be compiled.
                storage = {}  # an empty dict to keep things in
                setattr(instance, storage_name, storage)
                storage['free_vars'] = tf.placeholder(tf.float64, [None])
                instance.make_tf_array(storage['free_vars'])
                storage['tf_args'] = [tf.placeholder(*a) for a in self.tf_arg_tuples]
                with instance.tf_mode():
                    storage['tf_result'] = tf_method(instance, *storage['tf_args'])
                storage['session'] = tf.Session()
                storage['session'].run(tf.initialize_all_variables(), feed_dict=instance.get_feed_dict())
            feed_dict = dict(zip(storage['tf_args'], np_args))
            feed_dict[storage['free_vars']] = instance.get_free_state()
            feed_dict.update(instance.get_feed_dict())
            return storage['session'].run(storage['tf_result'], feed_dict=feed_dict)

        return runnable


class Parameterized(Parentable):
    """
    An object to contain parameters and data.

    This object is designed to be part of a tree, with Param and DataHolder
    objects at the leaves. We can then recurse down the tree to find all the
    parameters and data (leaves), or recurse up the tree (using highest_parent)
    from the leaves to the root.

    A useful application of such a recursion is 'tf_mode', where the parameters
    appear as their _tf_array variables. This allows us to build models on
    those parameters. During _tf_mode, the __getattribute__ method is
    overwritten to return tf arrays in place of parameters (and data).

    Another recursive function is build_prior which sums the log-prior from all
    of the tree's parameters (whilst in tf_mode!).
    """

    def __init__(self):
        Parentable.__init__(self)
        self._tf_mode = False

    def get_parameter_dict(self, d=None):
        if d is None:
            d = {}
        for p in self.sorted_params:
            p.get_parameter_dict(d)
        return d

    def set_parameter_dict(self, d):
        for p in self.sorted_params:
            p.set_parameter_dict(d)

    def get_samples_df(self, samples):
        """
        Given a numpy array where each row is a valid free-state vector, return
        a pandas.DataFrame which contains the parameter name and associated samples
        in the correct form (e.g. with positive constraints applied).
        """
        d = pd.DataFrame()
        for p in self.sorted_params:
            d = pd.concat([d, p.get_samples_df(samples)], axis=1)
        return d

    def __getattribute__(self, key):
        """
        Here, we overwrite the getattribute method.

        If tf mode is off, this does nothing.

        If tf mode is on, all child parameters will appear as their tf
        representations.
        """
        o = object.__getattribute__(self, key)
        if isinstance(o, (Param, DataHolder)) and object.__getattribute__(self, '_tf_mode'):
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

        # If we already have an atribute with that key, decide what to do:
        if key in self.__dict__.keys():
            p = getattr(self, key)

            # if the existing attribute is a parameter, and the value is an
            # array (or float, int), then set the _array of that parameter
            if isinstance(p, Param) and isinstance(value, (np.ndarray, float, int)):
                p._array[...] = value
                return  # don't call object.setattr or set the _parent value

            # if the existing attribute is a Param (or Parameterized), and the
            # new attribute is too, replace the attribute and set the model to
            # recompile if necessary.
            if isinstance(p, Param) and isinstance(value, (Param, Parameterized)):
                p._parent = None  # unlink the old Parameter from this tree
                if hasattr(self.highest_parent, '_needs_recompile'):
                    self.highest_parent._needs_recompile = True

            # if the existing atribute is a DataHolder, set the value of the data inside
            if isinstance(p, DataHolder) and isinstance(value, np.ndarray):
                p.set_data(value)
                return  # don't call object.setattr or set the _parent value

        # use the standard setattr
        object.__setattr__(self, key, value)

        # make sure a new child node knows this is the _parent:
        if isinstance(value, Parentable) and key is not '_parent':
            value._parent = self

        if key == '_needs_recompile':
            self._kill_autoflow()

    def _kill_autoflow(self):
        """Remove all AutoFlow storage dicts recursively.

        If AutoFlow functions become invalid, because recompilation is
        required, this function recurses the structure removing all AutoFlow
        dicts. Subsequent calls to to those functions will casue AutoFlow to regenerate.
        """
        for key in self.__dict__.keys():
            if key[0] == '_' and key[-11:] == '_AF_storage':
                delattr(self, key)
        [p._kill_autoflow() for p in self.sorted_params if isinstance(p, Parameterized)]

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

    def get_param_index(self, param_to_index):
        """
        Given a parameter, compute the position of that parameter on the
        free-state vector. This returns:
          - count: an integer representing the position
          - found: a bool representing whether the parameter was found.
        """
        found = False
        count = 0
        for p in self.sorted_params:
            if isinstance(p, Param):
                if p is param_to_index:
                    found = True
                    break
                else:
                    count += p.get_free_state().size
            elif isinstance(p, Parameterized):
                extra, found = p.get_param_index(param_to_index)
                count += extra
                if found:
                    break
        return count, found

    @property
    def sorted_params(self):
        """
        Return a list of all the child parameters, sorted by id. This makes
        sure they're always in the same order.
        """
        params = [child for key, child in self.__dict__.items()
                  if isinstance(child, (Param, Parameterized))
                  and key is not '_parent']
        return sorted(params, key=id)

    @property
    def data_holders(self):
        """
        Return a list of all the child DataHolders
        """
        return [child for key, child in self.__dict__.items()
                if isinstance(child, DataHolder)]

    @property
    def fixed(self):
        return all(p.fixed for p in self.sorted_params)

    @fixed.setter
    def fixed(self, val):
        for p in self.sorted_params:
            p.fixed = val

    def get_free_state(self):
        """
        recurse get_free_state on all child parameters, and hstack them.
        """
        # Here, additional empty array allows hstacking of empty list
        return np.hstack([p.get_free_state() for p in self.sorted_params]
                         + [np.empty(0)])

    def get_feed_dict(self):
        """
        Recursively fetch a dictionary matching up fixed-placeholders to
        associated values
        """
        d = {}
        for p in self.sorted_params + self.data_holders:
            d.update(p.get_feed_dict())
        return d

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
            # do tf stuff, like
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

        The idea is that in tf_mode, we can easily get references to the
        tf representation of parameters in order to construct tf
        objective functions.
        """
        self._begin_tf_mode()
        yield
        self._end_tf_mode()

    def _begin_tf_mode(self):
        [child._begin_tf_mode() for child in self.sorted_params
         if isinstance(child, Parameterized)]
        self._tf_mode = True

    def _end_tf_mode(self):
        [child._end_tf_mode() for child in self.sorted_params
         if isinstance(child, Parameterized)]
        self._tf_mode = False

    def build_prior(self):
        """
        Build a tf expression for the prior by summing all child-node priors.
        """
        return sum([p.build_prior() for p in self.sorted_params])

    def __str__(self, prepend=''):
        prepend += self.name + '.'
        return '\n'.join([p.__str__(prepend) for p in self.sorted_params])

    def _html_table_rows(self, name_prefix=''):  # pragma: no cover
        """
        Get the rows of the html table for this object
        """
        name_prefix += self.name + '.'
        return ''.join([p._html_table_rows(name_prefix)
                        for p in self.sorted_params])

    def _repr_html_(self):
        """
        Build a small html table for display in the jupyter notebook.
        """
        html = ["<table id='params' width=100%>"]

        # build the header
        header = "<tr>"
        header += "<td>Name</td>"
        header += "<td>values</td>"
        header += "<td>prior</td>"
        header += "<td>constraint</td>"
        header += "</tr>"
        html.append(header)

        html.append(self._html_table_rows())

        html.append("</table>")
        return ''.join(html)

    def __setstate__(self, d):
        Parentable.__setstate__(self, d)
        # reinstate _parent graph
        for p in self.sorted_params + self.data_holders:
            p._parent = self


class ParamList(Parameterized):
    """
    A list of parameters. This allows us to store parameters in a list whilst
    making them 'visible' to the GPflow machinery.

    The correct usage is

    >>> my_list = GPflow.param.ParamList([Param1, Param2])

    You can then iterate through the list. For example, to compute the sum:
    >>> my_sum = reduce(tf.add, my_list)

    or the sum of the squares:
    >>> rmse = tf.sqrt(reduce(tf.add, map(tf.square, my_list)))

    You can append things:
    >>> my_list.append(GPflow.kernels.RBF(1))

    but only if the are Parameters (or Parameterized objects). You can set the
    value of Parameters in the list:

    >>> my_list = GPflow.param.ParamList([GPflow.param.Param(2)])
    >>> print my_list
    unnamed.item0 transform:(none) prior:None
    [ 2.]
    >>> my_list[0] = 12
    >>> print my_list
    unnamed.item0 transform:(none) prior:None
    [ 12.]

    But you can't change elements of the list by assignment:
    >>> my_list = GPflow.param.ParamList([GPflow.param.Param(2)])
    >>> new_param = GPflow.param.Param(4)
    >>> my_list[0] = new_param # raises exception

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
        assert isinstance(item, (Param, Parameterized)), \
            "this object is for containing parameters"
        item._parent = self
        self.sorted_params.append(item)

    def __setitem__(self, key, value):
        """
        It's not possible to assign to things in the list, but it is possible
        to set their values by assignment.
        """
        self.sorted_params[key]._array[...] = value
