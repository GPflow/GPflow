# pylint: skip-file
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow interface for third-party optimizers."""

from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, gradients, variables
from tensorflow.python.platform import tf_logging as logging

__all__ = ['ExternalOptimizerInterface', 'ScipyOptimizerInterface']


class ExternalOptimizerInterface(object):
    """Base class for interfaces with external optimization algorithms.

    Subclass this and implement `_minimize` in order to wrap a new optimization
    algorithm.

    `ExternalOptimizerInterface` should not be instantiated directly; instead use
    e.g. `ScipyOptimizerInterface`.

    @@__init__

    @@minimize
    """

    def __init__(self, loss, var_list=None, equalities=None, inequalities=None, var_to_bounds=None,
                 **optimizer_kwargs):
        """Initialize a new interface instance.

        Args:
        loss: A scalar `Tensor` to be minimized.
        var_list: Optional `list` of `Variable` objects to update to minimize
            `loss`.  Defaults to the list of variables collected in the graph
            under the key `GraphKeys.TRAINABLE_VARIABLES`.
        equalities: Optional `list` of equality constraint scalar `Tensor`s to be
            held equal to zero.
        inequalities: Optional `list` of inequality constraint scalar `Tensor`s
            to be held nonnegative.
        var_to_bounds: Optional `dict` where each key is an optimization
            `Variable` and each corresponding value is a length-2 tuple of
            `(low, high)` bounds. Although enforcing this kind of simple constraint
            could be accomplished with the `inequalities` arg, not all optimization
            algorithms support general inequality constraints, e.g. L-BFGS-B. Both
            `low` and `high` can either be numbers or anything convertible to a
            NumPy array that can be broadcast to the shape of `var` (using
            `np.broadcast_to`). To indicate that there is no bound, use `None` (or
            `+/- np.infty`). For example, if `var` is a 2x3 matrix, then any of
            the following corresponding `bounds` could be supplied:
            * `(0, np.infty)`: Each element of `var` held positive.
            * `(-np.infty, [1, 2])`: First column less than 1, second column less
            than 2.
            * `(-np.infty, [[1], [2], [3]])`: First row less than 1, second row less
            than 2, etc.
            * `(-np.infty, [[1, 2, 3], [4, 5, 6]])`: Entry `var[0, 0]` less than 1,
            `var[0, 1]` less than 2, etc.
        **optimizer_kwargs: Other subclass-specific keyword arguments.
        """
        self.optimizer_kwargs = optimizer_kwargs

        self._loss = loss
        self._equalities = equalities or []
        self._inequalities = inequalities or []
        self._var_to_bounds = var_to_bounds

        if var_list is None:
            self._vars = variables.trainable_variables()
        elif var_list == []:
            raise ValueError("No variables to optimize.")
        else:
            self._vars = list(var_list)

        self._packed_bounds = []
        self._update_placeholders = []
        self._var_updates = []
        self._packed_var = None
        self._packed_loss_grad = None
        self._packed_equality_grads = []
        self._packed_inequality_grads = []
        self._var_shapes = None
        self._feed_dict = None

    def minimize(self,
                 session=None,
                 feed_dict=None,
                 fetches=None,
                 step_callback=None,
                 loss_callback=None,
                 **optimizer_kwargs):
        """Minimize a scalar `Tensor`.

        Variables subject to optimization are updated in-place at the end of
        optimization.

        Note that this method does *not* just return a minimization `Op`, unlike
        `Optimizer.minimize()`; instead it actually performs minimization by
        executing commands to control a `Session`.

        Args:
          session: A `Session` instance.
          feed_dict: A feed dict to be passed to calls to `session.run`.
          fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
            as positional arguments.
          step_callback: A function to be called at each optimization step;
            arguments are the current values of all optimization variables
            flattened into a single vector.
          loss_callback: A function to be called every time the loss and gradients
            are computed, with evaluated fetches supplied as positional arguments.
          **optimizer_kwargs: kwargs to pass to ScipyOptimizer `minimize` method.
        """
        session = session or ops.get_default_session()
        self.init_optimize(session=session, fetches=fetches, loss_callback=loss_callback, **optimizer_kwargs)
        self.optimize(session=session, step_callback=step_callback, feed_dict=feed_dict)

    def init_optimize(self, session=None, fetches=None, loss_callback=None, **optimizer_kwargs):
        """Create intermediate tensors for optimizing a scalar `Tensor`.

        Variables subject to optimization are updated in-place at the end of
        optimization.

        Note that this method does *not* just return a minimization `Op`, unlike
        `Optimizer.minimize()`; instead it actually performs minimization by
        executing commands to control a `Session`.

        Args:
          session: A `Session` instance.
          feed_dict: A feed dict to be passed to calls to `session.run`.
          fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
            as positional arguments.
          step_callback: A function to be called at each optimization step;
            arguments are the current values of all optimization variables
            flattened into a single vector.
          loss_callback: A function to be called every time the loss and gradients
            are computed, with evaluated fetches supplied as positional arguments.
          **optimizer_kwargs: kwargs to pass to ScipyOptimizer `minimize` method.
        """
        session = session or ops.get_default_session()
        fetches = fetches or []

        loss_callback = loss_callback or (lambda *fetches: None)

        # Get initial value from TF session.
        self._initialize_updated_shapes(session)
        self._make_minimize_tensors(session, fetches, loss_callback, **optimizer_kwargs)

    def optimize(self, session=None, step_callback=None, feed_dict=None):
        """
        Runs optimization on a scalar `Tensor` defined by `init_optimize`.

        Args:
          **run_kwargs: kwargs to pass to `session.run`.
        """
        session = session or ops.get_default_session()
        step_callback = step_callback or (lambda xk: None)

        # Perform minimization.
        self._feed_dict = feed_dict if feed_dict is not None else {}
        initial_packed_var_val = session.run(self._packed_var)
        packed_var_val = self._minimize(initial_val=initial_packed_var_val,
                                        packed_bounds=self._packed_bounds,
                                        optimizer_kwargs=self.optimizer_kwargs,
                                        step_callback=step_callback,
                                        **self._minimize_args)

        var_vals = [packed_var_val[packing_slice] for packing_slice in self._packing_slices]

        # Set optimization variables to their new values.
        run_feed_dict = dict(zip(self._update_placeholders, var_vals))
        session.run(self._var_updates, feed_dict=run_feed_dict)

    def _make_minimize_tensors(self, session, fetches, callback, **optimizer_kwargs):
        # Construct loss function and associated gradient.
        loss_grad_func = self._make_eval_func([self._loss, self._packed_loss_grad], session, fetches, callback)

        # Construct equality constraint functions and associated gradients.
        equality_funcs = self._make_eval_funcs(self._equalities, session, fetches)
        equality_grad_funcs = self._make_eval_funcs(self._packed_equality_grads, session, fetches)

        # Construct inequality constraint functions and associated gradients.
        inequality_funcs = self._make_eval_funcs(self._inequalities, session, fetches)
        inequality_grad_funcs = self._make_eval_funcs(self._packed_inequality_grads, session, fetches)
        kwargs = dict(loss_grad_func=loss_grad_func,
                      equality_funcs=equality_funcs,
                      equality_grad_funcs=equality_grad_funcs,
                      inequality_funcs=inequality_funcs,
                      inequality_grad_funcs=inequality_grad_funcs)
        kwargs.update(optimizer_kwargs)
        self._minimize_args = kwargs

    def _initialize_updated_shapes(self, session):
        shapes = array_ops.shape_n(self._vars)
        var_shapes = list(map(tuple, session.run(shapes)))

        if self._var_shapes is not None:
            new_old_shapes = zip(self._var_shapes, var_shapes)
            if all([old == new for old, new in new_old_shapes]):
                return

        self._var_shapes = var_shapes
        vars_and_shapes = zip(self._vars, self._var_shapes)
        vars_and_shapes_dict = dict(vars_and_shapes)

        packed_bounds = None
        if self._var_to_bounds is not None:
            left_packed_bounds = []
            right_packed_bounds = []
            for var, var_shape in vars_and_shapes:
                shape = list(var_shape)
                bounds = (-np.infty, np.infty)
                if var in var_to_bounds:
                    bounds = var_to_bounds[var]
                left_packed_bounds.extend(list(np.broadcast_to(bounds[0], shape).flat))
                right_packed_bounds.extend(list(np.broadcast_to(bounds[1], shape).flat))
            packed_bounds = list(zip(left_packed_bounds, right_packed_bounds))
        self._packed_bounds = packed_bounds

        self._update_placeholders = [array_ops.placeholder(var.dtype) for var in self._vars]
        self._var_updates = [
            var.assign(array_ops.reshape(placeholder, vars_and_shapes_dict[var]))
            for var, placeholder in zip(self._vars, self._update_placeholders)
        ]

        loss_grads = _compute_gradients(self._loss, self._vars)
        equalities_grads = [_compute_gradients(equality, self._vars) for equality in self._equalities]
        inequalities_grads = [_compute_gradients(inequality, self._vars) for inequality in self._inequalities]

        self._packed_var = self._pack(self._vars)
        self._packed_loss_grad = self._pack(loss_grads)
        self._packed_equality_grads = [self._pack(equality_grads) for equality_grads in equalities_grads]
        self._packed_inequality_grads = [self._pack(inequality_grads) for inequality_grads in inequalities_grads]

        dims = [_prod(vars_and_shapes_dict[var]) for var in self._vars]
        accumulated_dims = list(_accumulate(dims))
        self._packing_slices = [slice(start, end) for start, end in zip(accumulated_dims[:-1], accumulated_dims[1:])]

    def _minimize(self, initial_val, loss_grad_func, equality_funcs, equality_grad_funcs, inequality_funcs,
                  inequality_grad_funcs, packed_bounds, step_callback, optimizer_kwargs):
        """Wrapper for a particular optimization algorithm implementation.

        It would be appropriate for a subclass implementation of this method to
        raise `NotImplementedError` if unsupported arguments are passed: e.g. if an
        algorithm does not support constraints but `len(equality_funcs) > 0`.

        Args:
        initial_val: A NumPy vector of initial values.
        loss_grad_func: A function accepting a NumPy packed variable vector and
            returning two outputs, a loss value and the gradient of that loss with
            respect to the packed variable vector.
        equality_funcs: A list of functions each of which specifies a scalar
            quantity that an optimizer should hold exactly zero.
        equality_grad_funcs: A list of gradients of equality_funcs.
        inequality_funcs: A list of functions each of which specifies a scalar
            quantity that an optimizer should hold >= 0.
        inequality_grad_funcs: A list of gradients of inequality_funcs.
        packed_bounds: A list of bounds for each index, or `None`.
        step_callback: A callback function to execute at each optimization step,
            supplied with the current value of the packed variable vector.
        optimizer_kwargs: Other key-value arguments available to the optimizer.

        Returns:
        The optimal variable vector as a NumPy vector.
        """
        raise NotImplementedError('To use ExternalOptimizerInterface, subclass from it and implement '
                                  'the _minimize() method.')

    @classmethod
    def _pack(cls, tensors):
        """Pack a list of `Tensor`s into a single, flattened, rank-1 `Tensor`."""
        if not tensors:
            return None
        elif len(tensors) == 1:
            return array_ops.reshape(tensors[0], [-1])
        else:
            flattened = [array_ops.reshape(tensor, [-1]) for tensor in tensors]
            return array_ops.concat(flattened, 0)

    def _make_eval_func(self, tensors, session, fetches, callback=None):
        """Construct a function that evaluates a `Tensor` or list of `Tensor`s."""
        if not isinstance(tensors, list):
            tensors = [tensors]
        num_tensors = len(tensors)

        def eval_func(x):
            """Function to evaluate a `Tensor`."""
            shapes = dict(zip(self._vars, self._var_shapes))
            augmented_feed_dict = {
                var: x[packing_slice].reshape(shapes[var])
                for var, packing_slice in zip(self._vars, self._packing_slices)
            }
            augmented_feed_dict.update(self._feed_dict)
            augmented_fetches = tensors + fetches

            augmented_fetch_vals = session.run(augmented_fetches, feed_dict=augmented_feed_dict)

            if callable(callback):
                callback(*augmented_fetch_vals[num_tensors:])

            return augmented_fetch_vals[:num_tensors]

        return eval_func

    def _make_eval_funcs(self, tensors, session, fetches, callback=None):
        return [self._make_eval_func(tensor, session, fetches, callback) for tensor in tensors]


class ScipyOptimizerInterface(ExternalOptimizerInterface):
    """Wrapper allowing `scipy.optimize.minimize` to operate a `tf.Session`.

    Example:

    ```python
    vector = tf.Variable([7., 7.], 'vector')

    # Make vector norm as small as possible.
    loss = tf.reduce_sum(tf.square(vector))

    optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})

    with tf.Session() as session:
        optimizer.minimize(session)

    # The value of vector should now be [0., 0.].
    ```

    Example with simple bound constraints:

    ```python
    vector = tf.Variable([7., 7.], 'vector')

    # Make vector norm as small as possible.
    loss = tf.reduce_sum(tf.square(vector))

    optimizer = ScipyOptimizerInterface(
        loss, var_to_bounds={vector: ([1, 2], np.infty)})

    with tf.Session() as session:
        optimizer.minimize(session)

    # The value of vector should now be [1., 2.].
    ```

    Example with more complicated constraints:

    ```python
    vector = tf.Variable([7., 7.], 'vector')

    # Make vector norm as small as possible.
    loss = tf.reduce_sum(tf.square(vector))
    # Ensure the vector's y component is = 1.
    equalities = [vector[1] - 1.]
    # Ensure the vector's x component is >= 1.
    inequalities = [vector[0] - 1.]

    # Our default SciPy optimization algorithm, L-BFGS-B, does not support
    # general constraints. Thus we use SLSQP instead.
    optimizer = ScipyOptimizerInterface(
        loss, equalities=equalities, inequalities=inequalities, method='SLSQP')

    with tf.Session() as session:
        optimizer.minimize(session)

    # The value of vector should now be [1., 1.].
    ```
    """

    _DEFAULT_METHOD = 'L-BFGS-B'

    def _minimize(self, initial_val, loss_grad_func, equality_funcs, equality_grad_funcs, inequality_funcs,
                  inequality_grad_funcs, packed_bounds, step_callback, optimizer_kwargs):
        def loss_grad_func_wrapper(x):
            # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
            loss, gradient = loss_grad_func(x)
            return loss, gradient.astype('float64')

        method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

        constraints = []
        for func, grad_func in zip(equality_funcs, equality_grad_funcs):
            constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
        for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
            constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

        minimize_args = [loss_grad_func_wrapper, initial_val]
        minimize_kwargs = {
            'jac': True,
            'callback': step_callback,
            'method': method,
            'constraints': constraints,
            'bounds': packed_bounds,
        }

        for kwarg in minimize_kwargs:
            if kwarg in optimizer_kwargs:
                if kwarg == 'bounds':
                    # Special handling for 'bounds' kwarg since ability to specify bounds
                    # was added after this module was already publicly released.
                    raise ValueError('Bounds must be set using the var_to_bounds argument')
                raise ValueError('Optimizer keyword arg \'{}\' is set '
                                 'automatically and cannot be injected manually'.format(kwarg))

        minimize_kwargs.update(optimizer_kwargs)
        if method == 'SLSQP':
            # SLSQP doesn't support step callbacks. Obviate associated warning
            # message.
            del minimize_kwargs['callback']

        import scipy.optimize  # pylint: disable=g-import-not-at-top
        result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)
        logging.info(
            'Optimization terminated with:\n'
            '  Message: %s\n'
            '  Objective function value: %f\n'
            '  Number of iterations: %d\n'
            '  Number of functions evaluations: %d', result.message, result.fun, result.nit, result.nfev)

        return result['x']


def _accumulate(list_):
    total = 0
    yield total
    for x in list_:
        total += x
        yield total


def _prod(array):
    prod = 1
    for value in array:
        prod *= value
    return prod


def _compute_gradients(tensor, var_list):
    grads = gradients.gradients(tensor, var_list)
    # tf.gradients sometimes returns `None` when it should return 0.
    return [grad if grad is not None else array_ops.zeros_like(var) for var, grad in zip(var_list, grads)]
