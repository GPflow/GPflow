from __future__ import print_function
from .param import Param, Parameterized
from scipy.optimize import minimize
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
    _previous = None
    def __init__(self, objective):
        self._objective = objective
        self._previous_f = np.inf
    def __call__(self, x):
        f, g = self._objective(x)
        g_is_fin = np.isfinite(g)
        if np.all(g_is_fin):
            self._previous_x = x # store the last know good value
            self._previous_f = f
            return f, g
        else:
            print("Warning: inf or nan in gradient: replacing with zeros")
            return f, np.where(g_is_fin, g, 0.)

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

    @property
    def name(self):
        return self._name
    def _compile(self):
        """
        compile the tensorflow function "self._objective"
        """
        self.make_tf_array(self._free_vars)
        with self.tf_mode():
            f = self.build_likelihood() + self.build_prior()
            g, = tf.gradients(f, self._free_vars)

        minusF = tf.neg( f, name = 'objective' )
        minusG = tf.neg( g, name = 'grad_objective' )
        #initialize variables. I confess I don;t understand what this does - JH
        init = tf.initialize_all_variables()
        self._session.run(init)

        #build tensorflow functions for computing the likelihood and predictions
        print("compiling tensorflow function...")
        sys.stdout.flush()
        def obj(x):
            return self._session.run([minusF,minusG], feed_dict={self._free_vars: x})
        self._objective = obj
        print("done")
        sys.stdout.flush()
        self._needs_recompile = False

    def sample(self, num_samples, Lmax=20, epsilon=0.01, verbose=False):
        """
        use hybrid Monte Carlo to draw samples from the posterior of the model. 
        """
        if self._needs_recompile:
            self._compile()
        return hmc.sample_HMC(self._objective, num_samples, Lmax, epsilon, x0=self.get_free_state(), verbose=verbose)

    def optimize(self, method='L-BFGS-B', callback=None, max_iters=1000, **kw):
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
            self.set_state(obj._previous)
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
    def __init__(self, X, Y, kern, likelihood, mean_function, name='model'):
        self.X, self.Y, self.kern, self.likelihood, self.mean_function = X, Y, kern, likelihood, mean_function
        Model.__init__(self, name)

    def build_predict(self):
        raise NotImplementedError

    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points Xnew
        """
        self._compile()
        tf_Xnew = tf.placeholder('float64')
        with self.tf_mode():
            pred_f_mean, pred_f_var = self.build_predict(tf_Xnew)
        return self._session.run([pred_f_mean, pred_f_var],
                                 feed_dict={self._free_vars:self.get_free_state(), tf_Xnew:Xnew })

    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        tf_Xnew = tf.placeholder('float64')
        with self.tf_mode():
            pred_f_mean, pred_f_var = self.build_predict(tf_Xnew)
            pred_y_mean, pred_y_var = self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
        return self._session.run([pred_y_mean, pred_y_var],
                                 feed_dict={self._free_vars:self.get_free_state(), tf_Xnew:Xnew })

    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log denisty of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        tf_Xnew = tf.placeholder('float64')
        tf_Ynew = tf.placeholder('float64')
        with self.tf_mode():
            pred_f_mean, pred_f_var = self.build_predict(tf_Xnew)
            pred_y_density = self.likelihood.predict_density(pred_f_mean, pred_f_var, tf_Ynew)
        return self._session.run(pred_y_density,
                                 feed_dict={self._free_vars:self.get_free_state(),
                                            tf_Xnew:Xnew,
                                            tf_Ynew:Ynew })



            

