from __future__ import print_function
from .param import Param, Parameterized
from scipy.optimize import minimize, OptimizeResult
import numpy as np
import tensorflow as tf
import hmc
import sys

class ObjectiveWrapper(object):
    """
    A simple class to wrap the objective function in order to make it more robust. 

    The previosly seen state is cached so that we can easily acess it if the
    model crashes.
    """
    _previous_x = None
    def __init__(self, objective):
        self._objective = objective
    def __call__(self, x):
        f, g = self._objective(x)
        g_is_fin = np.isfinite(g)
        if np.all(g_is_fin):
            self._previous_x = x # store the last known good value
            return f, g
        else:
            print("Warning: inf or nan in gradient: replacing with zeros")
            return f, np.where(g_is_fin, g, 0.)


from functools import wraps
class AutoFlow:
    """
    This decorator-class is designed for use on methods of the Model class (below). 

    The idea is that methods that compute relevant quantities of the model
    (such as predictions) can define a tf graph which we automatically run when
    the (decorated) function is called.

    The syntax looks like:

    >>> class MyModel(Model):
    >>>   
    >>>   @AutoFlow(tf.placeholder(tf.float64), tf.placeholder(tf.float64))
    >>>   def my_method(self, x, y):
    >>>       #compute something, returning a tf graph.
    >>>       return tf.foo(self.baz, x, y)

    >>> m = MyModel()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> result = my_method(x, y)

    Now the output of the method call is the _result_ of the computation, equivalent to

    >>> m = MyModel()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> x_tf = tf.placeholder(tf.float64)
    >>> y_tf = tf.placeholder(tf.float64)
    >>> with m.tf_mode():
    >>>     graph = tf.foo(m.baz, x_tf, y_tf)
    >>> result = m._session.run(graph, feed_dict={x_tf:x, y_tf:y, m._free_vars:m.get_free_state()})

    Not only is the syntax cleaner, but multiple calls to the method will
    result in the graph being constructed only once. 

    """
    def __init__(self, *tf_args):
        self.tf_args = tf_args
    def __call__(self, tf_method):
        @wraps(tf_method)
        def runnable(instance, *np_args):
            graph_name = '_' + tf_method.__name__ + '_graph'
            if not hasattr(instance, graph_name):
                with instance.tf_mode():
                    instance.make_tf_array(instance._free_vars)
                    graph = tf_method(instance, *self.tf_args)
                    setattr(instance, graph_name, graph)
            feed_dict = dict(zip(self.tf_args, np_args))
            feed_dict[instance._free_vars] = instance.get_free_state()
            graph = getattr(instance, graph_name)
            return instance._session.run(graph, feed_dict=feed_dict)
        return runnable
    


class Model(Parameterized):
    """
    The Model base class. 

    To use this class, inherriting classes must define the method

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
    change, this object is notified and on optimization (or mcmc) the
    likelihood is recompiled. This allows fixing and constraining parameters,
    but only recompiling lazily.

    This object has a `_free_vars` tensorflow array. This array is ised to build
    the tensorflow representations of the Param objects during `make_tf_array`. 

    This object defines `optimize` and `sample` to allow for model fitting.
    """
    def __init__(self, name='model'):
        """
        name is a string describing this model.
        """
        Parameterized.__init__(self)
        self._name = name
        self._needs_recompile = True
        self._free_vars = tf.placeholder('float64', name='free_vars')
        self._session = tf.Session()
        self._free_vars = None

    @property
    def name(self):
        return self._name

    def _compile(self, optimizer=None):
        """
        compile the tensorflow function "self._objective"
        """
        # Make float32 hack
        float32_hack = False
        if optimizer is not None:
            if tf.float64 not in optimizer._valid_dtypes() and tf.float32 in optimizer._valid_dtypes():
                print("Using float32 hack for Tensorflow optimizers...")
                float32_hack = True

        self._free_vars = tf.Variable(self.get_free_state())
        if float32_hack:
            self._free_vars32 = tf.Variable(self.get_free_state().astype(np.float32))
            self._free_vars = tf.cast(self._free_vars32, tf.float64)

        self.make_tf_array(self._free_vars)
        with self.tf_mode():
            f = self.build_likelihood() + self.build_prior()
            g, = tf.gradients(f, self._free_vars)

        self._minusF = tf.neg( f, name = 'objective' )
        self._minusG = tf.neg( g, name = 'grad_objective' )

        # The optimiser needs to be part of the computational graph, and needs
        # to be initialised before tf.initialise_all_variables() is called.
        if optimizer is None:
            opt_step = None
        else:
            if float32_hack:
                opt_step = optimizer.minimize(tf.cast(self._minusF, tf.float32), var_list=[self._free_vars32])
            else:
                opt_step = optimizer.minimize(self._minusF, var_list=[self._free_vars])
        init = tf.initialize_all_variables()
        self._session.run(init)

        #build tensorflow functions for computing the likelihood and predictions
        print("compiling tensorflow function...")
        sys.stdout.flush()
        def obj(x):
            return self._session.run([self._minusF, self._minusG], feed_dict={self._free_vars: x})
        self._objective = obj
        print("done")
        sys.stdout.flush()
        self._needs_recompile = False

        return opt_step

    def __setattr__(self, key, value):
        Parameterized.__setattr__(self, key, value)
        #delete any AutoFlow related graphs
        if key=='_needs_recompile' and value:
            for key in filter(lambda x : x[0]=='_' and x[-6:]=='_graph', dir(self)):
                delattr(self, key)

    def sample(self, num_samples, Lmax=20, epsilon=0.01, verbose=False):
        """
        use hybrid Monte Carlo to draw samples from the posterior of the model. 
        """
        if self._needs_recompile:
            self._compile()
        return hmc.sample_HMC(self._objective, num_samples, Lmax, epsilon, x0=self.get_free_state(), verbose=verbose)

    def optimize(self, method='L-BFGS-B', callback=None, max_iters=1000, calc_feed_dict=None, **kw):
        if type(method) is str:
            return self._optimize_np(method, callback, max_iters, **kw)
        else:
            return self._optimize_tf(method, callback, max_iters, calc_feed_dict, **kw)

    def _optimize_tf(self, method, callback, max_iters, calc_feed_dict):
        opt_step = self._compile(optimizer=method)

        try:
            iteration = 0
            while iteration < max_iters:
                if calc_feed_dict is None:
                    feed_dict = {}
                else:
                    feed_dict = calc_feed_dict()
                self._session.run(opt_step, feed_dict=feed_dict)
                if callback is not None:
                    callback(self._session.run(self._free_vars))
                iteration += 1
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, setting model with most recent state.")
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

    def _optimize_np(self, method='L-BFGS-B', callback=None, max_iters=1000, **kw):
        """
        Optimize the model to find the maximum likelihood  or MAP point. Here
        we wrap `scipy.optimize.minimize`, any keyword arguments are passed
        through as `options`. 

        KeyboardInterrupts are caught and the model is set to the most recent
        value tried by the optimization routine. 

        LinAlgErrors are caught and the optimizer is 'coaxed' away from that
        region by returning a low likelihood value.

        This method returns the results of the call to optimize.minimize. 
        """
        if self._needs_recompile:
            self._compile()

        options=dict(display=True, max_iters=max_iters)
        options.update(kw)

        #LBFGS-B hcks. the options are different, annoyingly.
        if method  == 'L-BFGS-B':
            options['maxiter'] = max_iters
            del options['max_iters']
            options['disp'] = options['display']
            del options['display']

        #here's the actual cll to minimize. cathc keyboard errors as harmless.
        obj = ObjectiveWrapper(self._objective)
        try:
            result = minimize(fun=obj,
                        x0=self.get_free_state(),
                        method=method,
                        jac=True, # self._objective returns the objective and the jacobian.
                        tol=None, # TODO: tol??
                        callback=callback,
                        options=options)
        except (KeyboardInterrupt): # pragma: no cover
            print("Caught KeyboardInterrupt, setting model with most recent state.")
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
    inherriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.
    """
    def __init__(self, X, Y, kern, likelihood, mean_function, minibatch=None, name='model'):
        self.dX, self.dY, self.kern, self.likelihood, self.mean_function = X, Y, kern, likelihood, mean_function
        Model.__init__(self, name)

        self.X = tf.Variable(self.dX)
        self.Y = tf.Variable(self.dY)
        self.minibatch = minibatch

    def _compile(self, optimizer=None):
        """
        compile the tensorflow function "self._objective"
        """
        opt_step = Model._compile(self, optimizer)
        def obj(x):
            ss = np.random.permutation(len(self.dX))[:self.minibatch]
            # This is much faster than above, and for N >> minibatch, it doesn't make much difference. This actually
            # becomes the limit when N is around 10**6, which isn't uncommon when using SVI. But it's not good for
            # sanity when setting the minibatch to the total size of the dataset...
            # ss = np.random.randint(len(self.dX), size=self.minibatch)
            return self._session.run([self._minusF, self._minusG], feed_dict={self._free_vars: x,
                                                                              self.X: self.dX[ss, :],
                                                                              self.Y: self.dY[ss, :]})
        self._objective = obj
        return opt_step

    def optimize(self, method='L-BFGS-B', callback=None, max_iters=1000, **kw):
        def calc_feed_dict():
            ss = np.random.randint(len(self.dX), size=self.minibatch)
            return {self.X: self.dX[ss, :], self.Y: self.dY[ss, :]}

        return Model.optimize(self, method, callback, max_iters, calc_feed_dict, **kw)

    def build_predict(self):
        raise NotImplementedError

    @AutoFlow(tf.placeholder(tf.float64))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points Xnew
        """
        return self.build_predict(Xnew)

    @AutoFlow(tf.placeholder(tf.float64))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)


    @AutoFlow(tf.placeholder(tf.float64), tf.placeholder(tf.float64))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log denisty of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    @property
    def minibatch(self):
        if self._minibatch is None:
            return len(self.dX)
        else:
            return self._minibatch

    @minibatch.setter
    def minibatch(self, val):
        self._minibatch = val



            

