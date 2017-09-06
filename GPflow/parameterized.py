# Copyright 2016 James Hensman, Mark van der Wilk, Valentine Svensson, alexggmatthews, fujiisoup
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

from __future__ import print_function, absolute_import

import sys
import warnings
import numpy as np
import tensorflow as tf
import session_manager

from scipy.optimize import minimize, OptimizeResult

from . import hmc, session

from .base import ISessionOwner, Parentable
from .base import CompilableNode, Compiled
from .param import Param
from .autoflow import AutoFlow
from .mean_functions import Zero
from .misc import is_valid_param_value
from .misc import FLOAT_TYPE
from .misc import GPflowError
from ._settings import settings


class Parameterized(CompilableNode):
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

    *Scoping*
    Parameterized classes can define functions that operate on tf variables. To
    wrap those functions in tensorflow scopes, the names of the scoped
    fucntions are stored in self.scoped_keys (a list of strings). Those
    functions are then called inside a tensorflow scope.

    """

    def __init__(self, name=None):
        super(Parameterized, self).__init__(name=name)
        self.scoped_keys = []
        self._prior_tensor = None

    @property
    def params(self):
        for key, param in self.__dict__.items():
            if not key.startswith('_') and isinstance(param, (Param, Parameterized)):
                yield param

    @property
    def free_params(self):
        for param in self.params:
            if not param.fixed:
                yield param

    @property
    def prior_tensor(self):
        return self._prior_tensor

    def free_param_tensors(self, graph=None):
        if self.is_compiled_check_consistency(graph) is Compiled.NOT_COMPILED:
            raise GPflowError('')
        for param in self.free_params:
            yield param.param_tensor

    def is_compiled(self, graph=None):
        graph = self.verified_graph(graph)
        param_graphs = set([param.graph for param in self.params])

        if None in param_graphs and param_graphs.issubset([None, graph]):
            return Compiled.NOT_COMPILED
        elif graph not in param_graphs:
            return Compiled.NOT_COMPATIBLE_GRAPH
        return Compiled.COMPILED

    @property
    def graph(self):
        for param in self.params:
            if param.graph is not None:
                return param.graph
        return None

    def compile(self, graph=None):
        graph = self.verified_graph(graph)
        with self.compilation_context(graph):
            for param in self.params:
                param.compile(graph)
            self._prior_tensor = self._build_prior()

    @property
    def fixed(self):
        """A boolean attribute to determine if all the child parameters of this node are fixed"""
        for param in self.params:
            if not param.fixed:
                return False
        return True

    @fixed.setter
    def fixed(self, value):
        for param in self.params:
            param.fixed = value

    def set_fixed(self, value, graph=None):
        for param in self.params:
            param.set_fixed(value, graph=graph)

    # TODO(awav): # pylint: disable=W0511
    #def randomize(self, distributions={}, skipfixed=True):
    #    """
    #    Calls randomize on all parameters in model hierarchy.
    #    """
    #    for param in self.sorted_params:
    #        param.randomize(distributions, skipfixed)

    def _build_prior(self):
        """
        Build a tf expression for the prior by summing all child-parameter priors.
        """
        graph = tf.get_default_graph()
        if self.is_compiled_check_consistency(graph) is Compiled.NOT_COMPILED:
            raise GPflowError('Parameterized object is not compilied.')
        return tf.add_n([param.prior_tensor for param in self.params], name=self._prior_name)

    @property
    def _prior_name(self):
        return 'prior'

    def _update_param_attribute(self, key, value):
        attr = getattr(self, key)
        param_like = (Param, Parameterized)
        if not isinstance(key, param_like):
            raise ValueError('Param-like attribute expected in assignment.')
        if isinstance(value, param_like):
            if self.is_compiled_check_consistency(value.graph) is Compiled.COMPILED:
                raise GPflowError('Parameterized object is compiled.')
            attr.set_parent()
            attr.set_name()
            value.set_parent(self)
            value.set_name(key)
            object.__setattr__(self, key, value)
        # elif - DataHolder:
        elif isinstance(attr, Param) and is_valid_param_value(value):
            attr.assign(value)
        else:
            msg = '"{0}" type cannot be assigned to param-like attribute.'
            raise ValueError(msg.format(type(value)))

    def _html_table_rows(self, name_prefix=''):
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

        if key.startswith('_'):
            object.__setattr__(self, key, value)
            return

        param_like = (Param, Parameterized)
        if key in self.__dict__.keys():
            if isinstance(getattr(self, key), param_like):
                self._update_param_attribute(key, value)
                return
        if isinstance(value, param_like):
            if self.is_compiled_check_consistency(value.graph) is Compiled.COMPILED:
                raise GPflowError('Parameterized object is compiled.')
            value.set_name(key)
            value.set_parent(self)
        object.__setattr__(self, key, value)

# TODO(awav)
    def __getstate__(self):
        d = Parentable.__getstate__(self)
        # do not pickle autoflow
        for key in list(d.keys()):
            if key[0] == '_' and key[-11:] == '_AF_storage':
                d.pop(key)
        return d

# TODO(awav)
    def __setstate__(self, d):
        Parentable.__setstate__(self, d)
        # reinstate _parent graph
        for p in self.sorted_params + self.data_holders:
            p._parent = self

# TODO(awav)
    def __str__(self, prepend=''):
        prepend += self.name + '.'
        return '\n'.join([p.__str__(prepend) for p in self.sorted_params])

    #def get_parameter_dict(self, d=None):
    #    if d is None:
    #        d = {}
    #    for p in self.sorted_params:
    #        p.get_parameter_dict(d)
    #    return d

    #def set_parameter_dict(self, d):
    #    for p in self.sorted_params:
    #        p.set_parameter_dict(d)

    #def get_samples_df(self, samples):
    #    """
    #    Given a numpy array where each row is a valid free-state vector, return
    #    a pandas.DataFrame which contains the parameter name and associated samples
    #    in the correct form (e.g. with positive constraints applied).
    #    """
    #    d = pd.DataFrame()
    #    for p in self.sorted_params:
    #        d = pd.concat([d, p.get_samples_df(samples)], axis=1)
    #    return d

    #def _kill_autoflow(self):
    #    """
    #    Remove all compiled AutoFlow methods recursively.
    #    If AutoFlow functions become invalid, because recompilation is
    #    required, this function recurses the structure removing all AutoFlow
    #    dicts. Subsequent calls to to those functions will casue AutoFlow to regenerate.
    #    """
    #    for key in list(self.__dict__.keys()):
    #        if key[0] == '_' and key[-11:] == '_AF_storage':
    #            delattr(self, key)
    #    [p._kill_autoflow() for p in self.sorted_params if isinstance(p, Parameterized)]

    #def make_tf_array(self, X):
    #    """
    #    Distribute a flat tensorflow array amongst all the child parameter of this instance.
    #    X is a tensorflow placeholder. It gets passed to all the children of
    #    this class (that are Parameterized or Param objects), which then
    #    construct their tf_array variables from consecutive sections.
    #    """
    #    count = 0
    #    for dh in self.data_holders:
    #        dh.make_tf_array()
    #    for p in self.sorted_params:
    #        count += p.make_tf_array(X[count:])
    #    return count

    #def get_param_index(self, param_to_index):
   #    """
    #    Given a parameter, compute the position of that parameter on the free-state vector.
    #    This returns:
    #      - count: an integer representing the position
    #      - found: a bool representing whether the parameter was found.
    #    """
    #    found = False
    #    count = 0
    #    for p in self.sorted_params:
    #        if isinstance(p, Param):
    #            if p is param_to_index:
    #                found = True
    #                break
    #            else:
    #                count += p.get_free_state().size
    #        elif isinstance(p, Parameterized):
    #            extra, found = p.get_param_index(param_to_index)
    #            count += extra
    #            if found:
    #                break
    #    return count, found

    #@property
    #def sorted_params(self):
    #    """
    #    Return a list of all the child parameters, sorted by id.

    #    This makes sure they're always in the same order.
    #    """
    #    params = [child for key, child in self.__dict__.items()
    #              if isinstance(child, (Param, Parameterized)) and
    #              key is not '_parent']
    #    return sorted(params, key=lambda x: x.long_name)

    #@property
    #def data_holders(self):
    #    """
    #    Return a list of all the child DataHolders
    #    """
    #    return [child for key, child in self.__dict__.items()
    #            if isinstance(child, DataHolder)]

    #def __getattribute__(self, key):
    #    """
    #    Here, we overwrite the getattribute method.

    #    If tf mode is off, this does nothing.

    #    If tf mode is on, all child parameters will appear as their tf
    #    representations, and all functions that are designated in 'scoped_keys'
    #    will have aname scope applied.
    #    """
    #    o = object.__getattribute__(self, key)

    #    # if _tf_mode is False, or there is no _tf_mode, just return the object as normal.
    #    try:
    #        if not object.__getattribute__(self, '_tf_mode'):
    #            return o
    #    except AttributeError:
    #        return o

    #    # In tf_mode, if the object is a Param/Dataholder, ise the tf_array
    #    if isinstance(o, (Param, DataHolder)):
    #        return o._tf_array

    #    # in tf_mode, wrap functions is a scope
    #    elif key in object.__getattribute__(self, 'scoped_keys'):
    #        return NameScoped(self.long_name + '.' + key)(o)

    #    # finally, just return the object
    #    return o


class Model(Parameterized, ISessionOwner):
    """
    The Model base class.

    To use this class, inheriting classes must define the method

    >>>     build_likelihood(self)

    which returns a tensorflow representation of the model likelihood.

    Param and Parameterized objects that are children of the model can be used
    in the tensorflow expression. Children on the model are defined by simply
    doing:

    >>> m = Model()
    >>> p = Param(1.0)
    >>> m.p = p

    At compile time (i.e. when build_likelihood is called), the `Param` object
    becomes a tensorflow variable.

    The result of build_likelihood() is added to the prior (see Parameterized
    class) and the resulting objective and gradients are compiled into
    self._objective.

    This object has a `_needs_recompile` switch. When any of the child nodes
    change, this object is notified and on optimization (or MCMC) the
    likelihood is recompiled. This allows fixing and constraining parameters,
    but only recompiling lazily.

    This object has a `_free_vars` tensorflow array. This array is used to
    build the tensorflow representations of the Param objects during
    `make_tf_array`.

    This object defines `optimize` and `sample` to allow for model fitting.
    """

    class ObjectiveWrapper(object):
        """
        A simple class to wrap the objective function in order to make it more
        robust.

        The previously seen state is cached so that we can easily access it if the
        model crashes.
        """

        def __init__(self, objective):
            self._objective = objective
            self._previous_x = None

        def __call__(self, x):
            f, g = self._objective(x)
            g_is_fin = np.isfinite(g)
            if np.all(g_is_fin):
                self._previous_x = x  # store the last known good value
                return f, g
            else:
                print("Warning: inf or nan in gradient: replacing with zeros")
                return f, np.where(g_is_fin, g, 0.)

    def __init__(self, name=None, session=None):
        """
        name is a string describing this model.
        """
        super(Model, self).__init__(name=name)
        self._num_fevals = 0  # Keeps track of how often _objective is called
        self._session = session

    @property
    def graph(self):
        return self.session.graph

    @property
    def session(self):
        return self._session

    def compile(self, graph=None):
        """
        Compile the tensorflow function "self._objective".
        The `session` and `graph` parameters are mutually exclusive.
        :param session: TensorFlow Session. This parameter prevails `graph`
                        parameter. Custom created session will be used if
                        this argument is left default, i.e. None.
        :param graph: TensorFlow Graph. This argument ignored when `session`
                      differs from default value, otherwise it is passed to
                      new session constructor. Default TensorFlow graph value
                      is used, when `graph` equals None.
        :param optimizer: TensorFlow Optimizer.
        """

        if self.is_compiled_check_consistency(graph) is Compiled.COMPILED:
            return

        with self.compilation_context(graph):
            super(Model, self).compile(graph)

            func = tf.add(self.likelihood_tensor, self.prior_tensor)
            grad_func = tf.gradients(func, self.free_param_tensors(graph))

            objective = tf.negative(func, name='objective')
            objective_gradient = tf.negative(grad_func, name='gradient_objective')

            # The optimiser needs to be part of the computational graph,
            # and needs to be initialised before tf.initialise_all_variables()
            # is called.
            if optimizer is None:
                opt_step = None
            else:
                opt_step = optimizer.minimize(
                    self._minusF, var_list=[self._free_vars])
            init = tf.global_variables_initializer()

        session.run(init)
        self._session = session

        # build tensorflow functions for computing the likelihood
        if settings.verbosity.tf_compile_verb:
            print("compiling tensorflow function...")
        sys.stdout.flush()

        self._feed_dict_keys = self.get_feed_dict_keys()

        def obj(x):
            self.num_fevals += 1
            feed_dict = {self._free_vars: x}
            self.update_feed_dict(self._feed_dict_keys, feed_dict)
            f, g = self.session.run([self._minusF, self._minusG],
                                     feed_dict=feed_dict)
            return f.astype(np.float64), g.astype(np.float64)

        self._objective = obj
        if settings.verbosity.tf_compile_verb:
            print("done")
        sys.stdout.flush()
        self._needs_recompile = False

        return opt_step

    @AutoFlow()
    def compute_log_prior(self):
        """ Compute the log prior of the model (uses AutoFlow)"""
        return self.prior_tensor

    @AutoFlow()
    def compute_log_likelihood(self):
        """ Compute the log likelihood of the model (uses AutoFlow on ``self.build_likelihood()``)"""
        return self.likelihood_tensor

    def sample(self, num_samples, Lmin=5, Lmax=20, epsilon=0.01, thin=1, burn=0,
               verbose=False, return_logprobs=False, RNG=np.random.RandomState(0)):
        """
        Use Hamiltonian Monte Carlo to draw samples from the model posterior.
        """
        if self._needs_recompile:
            self._compile()
        return hmc.sample_HMC(self._objective, num_samples,
                              Lmin=Lmin, Lmax=Lmax, epsilon=epsilon, thin=thin, burn=burn,
                              x0=self.get_free_state(), verbose=verbose,
                              return_logprobs=return_logprobs, RNG=RNG)

    def optimize(self, method='L-BFGS-B', tol=None, callback=None,
                 maxiter=1000, **kw):
        """
        Optimize the model by maximizing the likelihood (possibly with the
        priors also) with respect to any free variables.

        method can be one of:
            a string, corresponding to a valid scipy.optimize.minimize string
            a tensorflow optimizer (e.g. tf.optimize.AdaGrad)

        The callback function is executed by passing the current value of
        self.get_free_state()

        tol is the tolerance passed to scipy.optimize.minimize (ignored
            for tensorflow optimizers)

        max_iters defines the maximum number of iterations

        In the case of the scipy optimization routines, any additional keyword
        arguments are passed through.

        KeyboardInterrupts are caught and the model is set to the most recent
        value tried by the optimization routine.

        This method returns the results of the call to optimize.minimize, or a
        similar object in the tensorflow case.
        """

        if type(method) is str:
            return self._optimize_np(method, tol, callback, maxiter, **kw)
        else:
            return self._optimize_tf(method, callback, maxiter, **kw)

    def _optimize_tf(self, method, callback, maxiter):
        """
        Optimize the model using a tensorflow optimizer. See self.optimize()
        """
        opt_step = self._compile(optimizer=method)
        feed_dict = {}

        try:
            iteration = 0
            while iteration < maxiter:
                self.update_feed_dict(self._feed_dict_keys, feed_dict)
                self._session.run(opt_step, feed_dict=feed_dict)
                self.num_fevals += 1
                if callback is not None:
                    callback(self._session.run(self._free_vars))
                iteration += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting model\
                  with most recent state.")
            self.set_state(self._session.run(self._free_vars))
            return None

        final_x = self._session.run(self._free_vars)
        self.set_state(final_x)
        fun, jac = self._objective(final_x)
        r = OptimizeResult(x=final_x,
                           success=True,
                           message="Finished iterations.",
                           fun=fun,
                           jac=jac,
                           status="Finished iterations.")
        return r

    def _optimize_np(self, method='L-BFGS-B', tol=None, callback=None,
                     maxiter=1000, **kw):
        """
        Optimize the model to find the maximum likelihood  or MAP point. Here
        we wrap `scipy.optimize.minimize`, any keyword arguments are passed
        through as `options`.

        method is a string (default 'L-BFGS-B') specifying the scipy
        optimization routine, one of
            - 'Powell'
            - 'CG'
            - 'BFGS'
            - 'Newton-CG'
            - 'L-BFGS-B'
            - 'TNC'
            - 'COBYLA'
            - 'SLSQP'
            - 'dogleg'
        tol is the tolerance to be passed to the optimization routine
        callback is callback function to be passed to the optimization routine
        max_iters is the maximum number of iterations (used in the options dict
            for the optimization routine)
        """
        if self._needs_recompile:
            self._compile()

        options = dict(disp=settings.verbosity.optimisation_verb, maxiter=maxiter)
        if 'max_iters' in kw:  # pragma: no cover
            options['maxiter'] = kw.pop('max_iters')
            warnings.warn("Use `maxiter` instead of deprecated `max_iters`.", np.VisibleDeprecationWarning)

        if 'display' in kw:  # pragma: no cover
            options['disp'] = kw.pop('display')
            warnings.warn("Use `disp` instead of deprecated `display`.", np.VisibleDeprecationWarning)

        options.update(kw)

        # here's the actual call to minimize. Catch keyboard errors as harmless.
        obj = self.ObjectiveWrapper(self._objective)
        try:
            result = minimize(fun=obj,
                              x0=self.get_free_state(),
                              method=method,
                              jac=True,
                              tol=tol,
                              callback=callback,
                              options=options)
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting \
                  model with most recent state.")
            self.set_state(obj._previous_x)
            return None

        if settings.verbosity.optimisation_verb:
            print("optimization terminated, setting model state")
        self.set_state(result.x)
        return result

    def __getstate__(self):
        """
        This method is necessary for pickling objects
        """
        state = Parameterized.__getstate__(self)
        keys = ['_session', '_free_vars', '_objective',
                '_minusF', '_minusG', '_feed_dict_keys']
        for key in keys:
            state.pop(key, None)
        return state

    def __setstate__(self, d):
        Parameterized.__setstate__(self, d)
        self._needs_recompile = True



class GPModel(Model):
    """
    A base class for Gaussian process models, that is, those of the form

    .. math::
       :nowrap:

       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \\end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.

    For handling another data (Xnew, Ynew), set the new value to self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(self, X, Y, kern, likelihood, mean_function, name='model'):
        Model.__init__(self, name)
        self.mean_function = mean_function or Zero()
        self.kern, self.likelihood = kern, likelihood

        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        self.X, self.Y = X, Y

    def build_predict(self, *args, **kwargs):
        raise NotImplementedError

    @AutoFlow((FLOAT_TYPE, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self.build_predict(Xnew)

    @AutoFlow((FLOAT_TYPE, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.build_predict(Xnew, full_cov=True)

    @AutoFlow((FLOAT_TYPE, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=FLOAT_TYPE) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=FLOAT_TYPE)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @AutoFlow((FLOAT_TYPE, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((FLOAT_TYPE, [None, None]), (FLOAT_TYPE, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

class ParamList(Parameterized):
    """
    A list of parameters.

    This allows us to store parameters in a list whilst making them 'visible'
    to the GPflow machinery. The correct usage is

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

    def __init__(self, list_of_params):
        Parameterized.__init__(self)
        assert isinstance(list_of_params, list)
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

    def __len__(self):
        return len(self._list)

    def __setitem__(self, key, value):
        """
        It's not possible to assign to things in the list, but it is possible
        to set their values by assignment.
        """
        self.sorted_params[key]._array[...] = value
