# Copyright 2016 James Hensman, alexggmatthews
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


from __future__ import division, print_function

import tensorflow as tf
import numpy as np

def sample(model, num_samples, epsilon, lmin=1, lmax=1, thin=1, burn=0):
    """
    A straight-forward HMC implementation. The mass matrix is assumed to be the
    identity.

    f is a python function that returns the energy and its gradient

      f(x) = E(x), dE(x)/dx

    we then generate samples from the distribution

      pi(x) = exp(-E(x))/Z

    - num_samples is the number of samples to generate.
    - Lmin, Lmax, epsilon are parameters of the HMC procedure to be tuned.
    - x0 is the starting position for the procedure.
    - verbose is a flag which turns on the display of the running accept ratio.
    - thin is an integer which specifies the thinning interval
    - burn is an integer which specifies how many initial samples to discard.
    - RNG is a random number generator
    - return_logprobs is a boolean indicating whether to return the log densities alongside the samples.

    The total number of iterations is given by

      burn + thin * num_samples

    The return shape is always num_samples x D.

    The leafrog (Verlet) integrator works by picking a random number of steps
    uniformly between Lmin and Lmax, and taking steps of length epsilon.
    """

    def logprob_and_grads():
        logprob = tf.negative(model.objective)
        grads = tf.gradients(logprob, model.trainable_variables)
        return logprob, grads

    xs = model.trainable_variables
    args = [thin, xs, epsilon, lmin, lmax, logprob_and_grads]

    if burn > 0:
        _burning(burn, *args)

    result = tf.map_fn(lambda _: _thinning(*args), np.arange(num_samples), dtypes=...)
    session.run(result, )
    return


def _assign_variables(variables, values):
    return _map(lambda var, value: var.assign(value), variables, values)


def _copy_variables(variables):
    # NOTE: read_value with control_dependencies does not guarantee that
    #       that expected value will be returned.
    # return _map(lambda v: v.read_value(), variables)
    return _map(lambda var: var + 0, variables)

def _init_ps(xs):
    return _map(lambda x: tf.random_normal(tf.shape(x), dtype=x.dtype), xs)


def _update_ps(ps, grads, epsilon, coeff=1):
    return _map(lambda p, grad: p + coeff * epsilon * grad, ps, grads)


def _leapfrog_step(xs, ps, epsilon, max_iterations,
                   logprob_and_grads_fn):
    def update_xs(ps_values):
        return _map(lambda x, p: x.assign_add(epsilon * p), xs, ps_values)

    def premature_check(grads):
        fins = _map(lambda grad: tf.reduce_all(tf.is_finite(grad)), grads)
        return tf.reduce_all(fins)

    def cond(iteration, stop, _ps, _xs):
        return (not stop) and (iteration < max_iterations)

    def body(iteration, _stop, ps, _xs):
        xs_new = update_xs(ps)
        with tf.control_dependencies(xs_new):
            _, grads = logprob_and_grads_fn()
            proceed = premature_check(grads)
            ps_new = tf.cond(proceed, _update_ps(ps, grads, epsilon), ps)
            return iteration + 1, proceed, ps_new, xs_new

    i = tf.convert_to_tensor(0, dtype=tf.int32)
    stop = tf.convert_to_tensor(False, dtype=tf.bool)
    result = tf.while_loop(cond, body,
                           [i, stop, ps, xs],
                           parallel_iterations=1,
                           back_prop=False)
    with tf.control_dependencies(result):
        logprob_out, grads_out = logprob_and_grads_fn()
        _i, premature, ps_out, _xs  = result
        return premature, ps_out, logprob_out, grads_out


def _premature_reject(xs, xs_prev, logprob_and_grads):
    xs_new = _copy_variables(_assign_variables(xs, xs_prev))
    with tf.control_dependencies(xs_new):
        logprob, _ = logprob_and_grads()
        return xs_new, logprob

def _thinning(thin, xs, epsilon, lmin, lmax, logprob_and_grads_fn):
    def cond(i, _sample, _logprob):
        return i < thin

    def body(i, _xs, logprob_prev):
        xs_copy = _copy_variables(xs)
        with tf.control_dependencies(xs_copy):
            _, grads = logprob_and_grads_fn()

            ps_init = _init_ps(xs)
            ps = _update_ps(ps_init, grads, epsilon, coeff=+0.5)
            max_iters = tf.random_uniform(lmin, lmax, dtype=tf.int32)

            leapfrog_result = _leapfrog_step(xs, ps, epsilon, max_iters, logprob_and_grads_fn)
            reject, ps_new, logprob_new, grads_new = leapfrog_result

            xs_out, logprob_out = tf.cond(
                reject,
                _premature_reject(xs, xs_copy, logprob_prev),
                _reject_accept_proposal(
                    xs, xs_copy,
                    ps_new, ps_init,
                    logprob_new, logprob_prev,
                    grads_new, epsilon))

            return i + 1, xs_out, logprob_out

    i = tf.convert_to_tensor(0, dtype=tf.int32)
    logprob, _grads = logprob_and_grads_fn()
    _, xs_sample, logprob = tf.while_loop(cond, body,
                                          [i, xs, logprob],
                                          back_prop=False,
                                          parallel_iterations=1)
    return xs_sample, logprob

# work out whether to accept the proposal
def _reject_accept_proposal(xs, xs_prev,
                            ps, ps_prev,
                            logprob, logprob_prev,
                            grads, epsilon):
    ps = _update_ps(ps, grads, epsilon, coeff=-0.5)

    def dot(ps_values):
        return tf.reduce_sum(_map(lambda p: tf.reduce_sum(tf.square(p)), ps_values))

    log_accept_ratio = logprob - 0.5 * dot(ps) - logprob_prev + 0.5 * dot(ps_prev)
    logu = tf.log(tf.random_normal(shape=tf.shape(logprob)))

    def accept():
        return _copy_variables(xs), logprob

    def reject():
        return _copy_variables(_assign_variables(xs, xs_prev)), logprob_prev

    return tf.cond(logu < log_accept_ratio, accept(), reject())


def _map(func, *args, **kwargs):
    return [func(*a, **kwargs) for a in zip(*args)]
