# Copyright 2016 James Hensman, Mark van der Wilk,
#                Valentine Svensson, alexggmatthews,
#                PabloLeon, fujiisoup
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


from __future__ import absolute_import

import enum
import numpy as np
import tensorflow as tf
import pandas as pd

from .base import IPrior, ITransform, ICompilable, IGraphOwner
from .base import Parentable
from .transforms import Identity

from .misc import GPflowError
from .misc import is_number, is_tensorflow_variable, norm_dtype
from .misc import add_to_trainables, remove_from_trainables
from .misc import tensor_name, get_tensor_by_name

from ._settings import settings

FLOAT_TYPE = settings.dtypes.float_type


class Param(Parentable, ICompilable):
    """
    An object to represent parameters.

    **Getting and setting values**

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

    **Unconstrained optimization**

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


    **Fixes**


    There is a self.fixed flag, in which case the parameter does not get
    optimized. To enable this, during make_tf_array, a fixed parameter will be
    ignored, and a placeholder added to the feed_dict instead.

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


    **Compiling into tensorflow**

    The method

    >>> self.make_tf_array

    constructs a tensorflow representation of the parameter, from a tensorflow
    vector representing the free state. In this case, the parameter is
    represented as part of the 'free-state' vector associated with a model.
    However, if the parameters is fixed, then a placeholder is returned during
    a call to update_feed_dict, and the parameter is represented that way instead.

    **Priors and transforms**

    The `self.prior` object is used to place priors on parameters, and the
    `self.transform` object is used to enable unconstrained optimization and
    MCMC.
    """

    class ParamAttribute(enum.Enum):
        """
        When one of these attributes is set, notify a recompilation.
        """
        PRIOR = 'prior'
        TRANSFORM = 'transform'
        FIXED = 'fixed'

        @property
        def interface(self):
            if self.value == self.PRIOR.value:
                return IPrior
            elif self.value == self.TRANSFORM.value:
                return ITransform
            return None

    def __init__(self, value, transform=Identity()):
        if value is None:
            raise ValueError('The value must be either tensorflow variable, array or scalar.')
        Parentable.__init__(self)

        self._tensor = None
        self._externally_defined = False
        self._initial_value = None

        self.fixed = False
        self.prior = None
        self.transform = transform

        if is_tensorflow_variable(value):
            self._externally_defined = True
            self._tensor = value
            return

        if is_number(value):
            value = np.array(value, dtype=dtype)

        self._initial_value = value

    @property
    def shape(self):
        if self._tensor is not None:
            return self._tensor.shape
        return self._initial_value.shape

    @property
    def value(self):
        if self.is_compiled and not self.fixed:
            return self.root.session.run(self._tensor)
        return self._initial_value

    #def get_samples_df(self, samples):
    #    """
    #    Given a numpy array where each row is a valid free-state vector, return
    #    a pandas.DataFrame which contains the parameter name and associated samples
    #    in the correct form (e.g. with positive constraints applied).
    #    """
    #    if self.fixed:
    #        return pd.Series([self.value for _ in range(samples.shape[0])], name=self.full_name)
    #    start, _ = self.root.get_param_index(self)
    #    free_state_size = self.transform.free_state_size(self.shape)
    #    end = start + free_state_size
    #    samples = samples[:, start:end]
    #    samples = [np.atleast_1d(self.transform.forward(s).reshape(self.shape))
    #               for s in samples]
    #    return pd.Series(samples, name=self.full_name)

    #def compile(self, free_array):
    #    """
    #    free_array is a tensorflow vector which will be the optimisation
    #    target, i.e. it will be free to take any value.
    #    Here we take that array, and transform and reshape it so that it can be
    #    used to represent this parameter
    #    Then we return the number of elements that we've used to construct the
    #    array, so that it can be sliced for the next Param.
    #    """
    #    if self.fixed:
    #        # fixed parameters are treated by tf.placeholder
    #        self._tf_array = tf.placeholder(dtype=FLOAT_TYPE,
    #                                        shape=self._array.shape,
    #                                        name=self.name)
    #        # do not consider log jacobian for parameters that are fixed.
    #        self._log_jacobian = 0.0
    #        return 0
    #    free_size = self.transform.free_state_size(self.shape)
    #    x_free = free_array[:free_size]
    #    mapped_array = self.transform.tf_forward(x_free)
    #    self._tf_array = tf.reshape(mapped_array, self.shape)
    #    self._log_jacobian = self.transform.tf_log_jacobian(x_free)
    #    return free_size

    def set_state(self, x):
        """
        Given a vector x representing the 'free' state of this Param, transform
        it 'forwards' and store the result in self._array. The values in
        self._array can be accessed using self.value

        This is a numpy method.
        """
        if self.fixed:
            return 0
        free_size = self.transform.free_state_size(self.shape)
        new_array = self.transform.forward(x[:free_size]).reshape(self.shape)
        assert new_array.shape == self.shape
        self._array[...] = new_array
        return free_size

    def randomize(self, distributions={}, skipfixed=True):
        """
        Randomly assign the parameter a new value by sampling either from a
        provided distribution from GPflow.priors, the parameter's prior, or
        by using a default scheme where a standard normal variable is
        propagated through the parameters transform.
        Will not change fixed parameters unless skipfixed flag is set to False.

        Optional Input:
            distributions (dictionary) - a list of priors indexed by parameters.
                Defaults to an empty dictionary.
            skipfixed (boolean) - if True, parameter cannot be randomized.
                Defaults to True.
        """
        if not (skipfixed and self.fixed):
            if self in distributions.keys():
                self._array = distributions[self].sample(self.shape)
            else:
                try:
                    self._array = self.prior.sample(self.shape)
                except AttributeError:
                    randn = np.random.randn(
                        self.transform.free_state_size(self.shape))
                    self._array = self.transform.forward(randn).reshape(self.shape)

    @property
    def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return self._array.size

    @property
    def shape(self):
        """The shape of this parameter, equivalent to self.value.shape"""
        return self._array.shape

    @property
    def value_tensor(self):
        self._tensor

    @property
    def prior_tensor(self):
        self._get_tensor_by_name('prior')

    @property
    def is_compiled(self):
        if not isinstance(self.root, IGraphOwner):
            return False
        return self.root.is_compiled

    def compile(self, graph=None):
        """
        Compile a tensorflow representation of the parameter.
        """
        if self.is_compiled:
            try:
                get_tensor_by_name(self._tensor.name, graph=graph)
                return
            except GPflowError:
                raise GPflowError('Attempt to run compile with different graph.')

        graph = self.root.session.graph if graph is None else graph
        if not self._externally_defined:
            self._tensor = self._build_tensor(graph)
        self._build_prior(graph)

    def _build_tensor(self, graph=None):
        if graph is None:
            raise ValueError('The graph argument cannot be empty.')

        with graph.as_default():
            init = tf.constant_initializer(self._initial_value, dtype=FLOAT_TYPE)
            return tf.get_variable(self.full_name, initializer=init, dtype=FLOAT_TYPE)

    def _build_prior(self, graph=None):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if graph is None:
            raise ValueError('The graph argument cannot be empty.')

        if not is_tensorflow_variable(self._tensor):  # pragma: no cover
            raise GPflowError('Tensorflow array has not been compiled.')

        prior_name = self._prior_name()

        if self.prior is None:
            return tf.constant(0.0, FLOAT_TYPE, name=prior_name)

        var = self._tensor
        log_jacobian = self.transform.tf_log_jacobian(var)
        transformed_var = self.transform.tf_forward(self._tensor)
        logp_var = self.prior.logp(transformed_var)
        return tf.add(logp_var, log_jacobian, name=prior_name)

    def _prior_name(self):
        return tensor_name(self.full_name, 'prior')

    def _set_fixed(self, value):
        if not isinstance(value, bool):
            raise TypeError('Fixed property value must be boolean.')

        if self._externally_defined:
            raise GPflowError('Externally defined parameter is not modifiable.')

        prev_fixed = self.fixed

        if not self.is_compiled or prev_fixed == value:
            return

        object.__setattr__(self, 'fixed', value)
        graph = self.root.graph
        if value:
            remove_from_trainables(self._tensor, graph)
        else:
            add_to_trainables(self._tensor, graph)

    def _set_parameter_attribute(self, attr, value):
        if attr is self.ParamAttribute.FIXED:
            self._set_fixed(value)
            return

        if self.is_compiled:
            raise GPflowError('Parameter has already been compiled.')

        key = attr.value
        if not isinstance(key, attr.interface):
            msg = 'Property object "{0}" must implement interface "{1}".'
            raise GPflowError(msg.format(key, attr.interface))
        object.__setattr__(self, key, value)

    def _get_tensor_by_name(self, name):
        if self.is_compiled:
            raise GPflowError('Parameter is not compiled.')
        graph = self.root.session.graph
        return get_tensor_by_name(name, graph=graph)

    def _html_table_rows(self, name_prefix=''):
        """
        Construct a row of an html table, to be used in the jupyter notebook.
        """
        html = "<tr>"
        html += "<td>{0}</td>".format(name_prefix + self.name)
        html += "<td>{0}</td>".format(str(self._array).replace('\n', '</br>'))
        html += "<td>{0}</td>".format(str(self.prior))
        html += "<td>{0}</td>".format('[FIXED]' if self.fixed
                                      else str(self._transform))
        html += "</tr>"
        return html

    def __getstate__(self):
        d = Parentable.__getstate__(self)
        for key in ['_tensor']:
            d.pop(key, None)
        return d

    def __setstate__(self, d):
        Parentable.__setstate__(self, d)
        self._fixed = self.fixed  # make self._tf_array if the parameter is fixed
        # NB the parent property will be set by the parent object, apart from
        # for the top level, where it muct be None
        # the tf_array and _log jacobian will be replaced when the model is recompiled

    def __setattr__(self, key, value):
        """
        When some attributes are set, we need to recompile the tf model before
        evaluation.
        """
        try:
            attr = self.ParamAttribute(key)
            self._set_parameter_attribute(attr, value)
            return
        except ValueError:
            pass
        object.__setattr__(self, key, value)

    def __str__(self, prepend=''):
        return prepend + \
               '\033[1m' + self.name + '\033[0m' + \
               ' transform:' + str(self._transform) + \
               ' prior:' + str(self._prior) + \
               (' [FIXED]' if self.fixed else '') + \
               '\n' + str(self.value)


class DataHolder(Parentable, ICompilable):
    """
    An object to represent data which needs to be passed to tensorflow for computation.

    This behaves in much the same way as a Param (above), but is always
    'fixed'. On a call to update_feed_dict, a placeholder-numpy pair is added to the feed_dict.

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
        dtype = norm_dtype(array)
        self._array = np.asarray(array, dtype=dtype)
        assert on_shape_change in ['raise', 'pass', 'recompile']
        self.on_shape_change = on_shape_change

    def __getstate__(self):
        d = Parentable.__getstate__(self)
        try:
            d.pop('_tensor')
        except KeyError:
            pass
        return d

    def make_tf_array(self):
        self._tf_array = tf.placeholder(dtype=self._get_type(self._array),
                                        shape=[None] * self._array.ndim,
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
                self.root._needs_recompile = True
            elif self.on_shape_change == 'pass':
                self._array = array.copy()
            else:
                raise ValueError('invalid option')  # pragma: no cover

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
