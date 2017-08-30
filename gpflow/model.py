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
from .param import Parameterized, AutoFlow, DataHolder
from .mean_functions import Zero
from scipy.optimize import minimize, OptimizeResult
import numpy as np
import tensorflow as tf
from . import hmc
from . import session as session_mngr
from ._settings import settings
import sys

float_type = settings.dtypes.float_type


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


class Model(Parameterized):
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

    def __init__(self, name='model'):
        """
        name is a string describing this model.
        """
        Parameterized.__init__(self)
        self.scoped_keys.extend(['build_likelihood', 'build_prior'])
        self._name = name
        self._needs_recompile = True

        self.num_fevals = 0  # Keeps track of how often _objective is called
        self._session = None

    @property
    def name(self):
        return self._name

    @property
    def session(self):
        return self._session

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

    def compile(self, session=None, graph=None, optimizer=None):
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

        out_filename = settings.profiling.output_file_name + "_objective"

        default_session = tf.get_default_session()
        if session is None:
            if graph is None or (default_session is not None and
                                 default_session.graph is graph):
                session = default_session
        if session is None:
            session = session_mngr.get_session(
                graph=graph, output_file_name=out_filename)

        with session.graph.as_default():
            self._free_vars = tf.Variable(self.get_free_state())

            self.make_tf_array(self._free_vars)
            with self.tf_mode():
                f = self.build_likelihood() + self.build_prior()
                g = tf.gradients(f, self._free_vars)[0]

            self._minusF = tf.negative(f, name='objective')
            self._minusG = tf.negative(g, name='grad_objective')

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
        return self.build_prior()

    @AutoFlow()
    def compute_log_likelihood(self):
        """ Compute the log likelihood of the model (uses AutoFlow on ``self.build_likelihood()``)"""
        return self.build_likelihood()

    def sample(self, num_samples, Lmin=5, Lmax=20, epsilon=0.01, thin=1,
               burn=0, verbose=False, return_logprobs=False,
               RNG=np.random.RandomState(0)):
        """
        Use Hamiltonian Monte Carlo to draw samples from the model posterior.
        """
        if self._needs_recompile:
            self.compile()
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
        return self._optimize_tf(method, callback, maxiter, **kw)

    def _optimize_tf(self, method, callback, maxiter):
        """
        Optimize the model using a tensorflow optimizer. See self.optimize()
        """
        opt_step = self.compile(optimizer=method)
        feed_dict = {}

        try:
            iteration = 0
            while iteration < maxiter:
                self.update_feed_dict(self._feed_dict_keys, feed_dict)
                self.session.run(opt_step, feed_dict=feed_dict)
                self.num_fevals += 1
                if callback is not None:
                    callback(self.session.run(self._free_vars))
                iteration += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting model\
                  with most recent state.")
            self.set_state(self.session.run(self._free_vars))
            return None

        final_x = self.session.run(self._free_vars)
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
            self.compile()

        options = dict(disp=settings.verbosity.optimisation_verb, maxiter=maxiter)
        if 'max_iters' in kw:  # pragma: no cover
            options['maxiter'] = kw.pop('max_iters')
            import warnings
            warnings.warn("Use `maxiter` instead of deprecated `max_iters`.", np.VisibleDeprecationWarning)

        if 'display' in kw:  # pragma: no cover
            options['disp'] = kw.pop('display')
            import warnings
            warnings.warn("Use `disp` instead of deprecated `display`.", np.VisibleDeprecationWarning)

        options.update(kw)

        # here's the actual call to minimize. Catch keyboard errors as harmless.
        obj = ObjectiveWrapper(self._objective)
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

        likelihood._check_targets(Y.value)
        self.X, self.Y = X, Y
        self._session = None

    def build_predict(self, *args, **kwargs):
        raise NotImplementedError

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.build_predict(Xnew, full_cov=True)

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)
