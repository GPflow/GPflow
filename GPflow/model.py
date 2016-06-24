from __future__ import print_function
from .param import Parameterized, AutoFlow
from scipy.optimize import minimize, OptimizeResult
import numpy as np
import tensorflow as tf
from . import hmc
import sys


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
        self._name = name
        self._needs_recompile = True
        self._session = tf.Session()
        self._free_vars = tf.placeholder(tf.float64)
        self._data_dict = {}

    @property
    def name(self):
        return self._name

    def __getstate__(self):
        """
        This mehtod is necessary for pickling objects
        """
        d = Parameterized.__getstate__(self)
        d.pop('_session')
        d.pop('_free_vars')
        try:
            d.pop('_objective')
            d.pop('_minusF')
            d.pop('_minusG')
        except:
            pass
        return d

    def __setstate__(self, d):
        Parameterized.__setstate__(self, d)
        self._needs_recompile = True
        self._session = tf.Session()

    def get_feed_dict(self):
        """
        Return a dicitonary containing all the placeholder-value pairs that
        should be fed to tensorflow in order to evaluate the model
        """
        d = Parameterized.get_feed_dict(self)
        d.update(self._data_dict)
        return d

    def _compile(self, optimizer=None):
        """
        compile the tensorflow function "self._objective"
        """
        self._free_vars = tf.Variable(self.get_free_state())

        self.make_tf_array(self._free_vars)
        with self.tf_mode():
            f = self.build_likelihood() + self.build_prior()
            g, = tf.gradients(f, self._free_vars)

        self._minusF = tf.neg(f, name='objective')
        self._minusG = tf.neg(g, name='grad_objective')

        # The optimiser needs to be part of the computational graph, and needs
        # to be initialised before tf.initialise_all_variables() is called.
        if optimizer is None:
            opt_step = None
        else:
            opt_step = optimizer.minimize(self._minusF,
                                          var_list=[self._free_vars])
        init = tf.initialize_all_variables()
        self._session.run(init)

        # build tensorflow functions for computing the likelihood
        print("compiling tensorflow function...")
        sys.stdout.flush()

        def obj(x):
            feed_dict = {self._free_vars: x}
            feed_dict.update(self.get_feed_dict())
            return self._session.run([self._minusF, self._minusG],
                                     feed_dict=feed_dict)

        self._objective = obj
        print("done")
        sys.stdout.flush()
        self._needs_recompile = False

        return opt_step

    @AutoFlow()
    def compute_log_prior(self):
        return self.build_prior()

    @AutoFlow()
    def compute_log_likelihood(self):
        return self.build_likelihood()

    def sample(self, num_samples, Lmax=20, epsilon=0.01, verbose=False):
        """
        Use Hamiltonian Monte Carlo to draw samples from the model posterior.
        """
        if self._needs_recompile:
            self._compile()
        return hmc.sample_HMC(self._objective, num_samples,
                              Lmax, epsilon,
                              x0=self.get_free_state(), verbose=verbose)

    def optimize(self, method='L-BFGS-B', tol=None, callback=None,
                 max_iters=1000, calc_feed_dict=None, **kw):
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

        calc_feed_dict is an optional function which returns a dictionary
        (suitable for tf.Session.run's feed_dict argument)

        In the case of the scipy optimization routines, any additional keyword
        arguments are passed through.

        KeyboardInterrupts are caught and the model is set to the most recent
        value tried by the optimization routine.

        This method returns the results of the call to optimize.minimize, or a
        similar object in the tensorflow case.
        """

        if type(method) is str:
            return self._optimize_np(method, tol, callback, max_iters, **kw)
        else:
            return self._optimize_tf(method, callback, max_iters,
                                     calc_feed_dict, **kw)

    def _optimize_tf(self, method, callback, max_iters, calc_feed_dict):
        """
        Optimize the model using a tensorflow optimizer. See self.optimize()
        """
        opt_step = self._compile(optimizer=method)

        try:
            iteration = 0
            while iteration < max_iters:
                feed_dict = self.get_feed_dict()
                if calc_feed_dict is not None:
                    feed_dict.update(calc_feed_dict())
                self._session.run(opt_step, feed_dict=feed_dict)
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
                     max_iters=1000, **kw):
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

        options = dict(display=True, max_iters=max_iters)
        options.update(kw)

        # LBFGS-B hacks. the options are different, annoyingly.
        if method == 'L-BFGS-B':
            options['maxiter'] = max_iters
            del options['max_iters']
            options['disp'] = options['display']
            del options['display']

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
        except (KeyboardInterrupt):
            print("Caught KeyboardInterrupt, setting \
                  model with most recent state.")
            self.set_state(obj._previous_x)
            return None

        print("optimization terminated, setting model state")
        self.set_state(result.x)
        return result


class GPModel(Model):
    """
    A base class for Gaussian process models, that is, those of the form

       theta ~ p(theta)
       f ~ GP(m(x), k(x, x'; theta))
       F = f(X)
       Y|F ~ p(Y|F)

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.
    """

    def __init__(self, X, Y, kern, likelihood, mean_function, name='model'):
        self.kern, self.likelihood, self.mean_function = \
            kern, likelihood, mean_function
        Model.__init__(self, name)

        # set of data is stored in dict self._data_dict
        # self._data_dict will be feeded to tensorflow at the runtime.
        self.X = tf.placeholder(tf.float64, shape=X.shape, name="X")
        self.Y = tf.placeholder(tf.float64, shape=Y.shape, name="Y")
        self._data_dict = {self.X: X, self.Y: Y}

    def build_predict(self):
        raise NotImplementedError

    @AutoFlow((tf.float64, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self.build_predict(Xnew)

    @AutoFlow((tf.float64, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.build_predict(Xnew, full_cov=True)

    @AutoFlow((tf.float64, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict(Xnew, full_cov=True)
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i])
            shape = tf.pack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=tf.float64)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.pack(samples))

    @AutoFlow((tf.float64, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((tf.float64, [None, None]), (tf.float64, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    def update_data(self, X, Y):
        """
        Update data.
        If size of data was changed, then it will recompile.
        """
        if (X.shape != self._data_dict[self.X].shape) or \
           (Y.shape != self._data_dict[self.Y].shape):
            self.X = tf.placeholder(tf.float64, shape=X.shape, name="X")
            self.Y = tf.placeholder(tf.float64, shape=Y.shape, name="Y")
            # raise the recompilation flag
            self._needs_recompile = True
            # autoflow also should be killed.
            self._kill_autoflow()

        self._data_dict = {self.X: X, self.Y: Y}
