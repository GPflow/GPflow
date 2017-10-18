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


def sample_HMC(f, num_samples, Lmin, Lmax, epsilon, x0, verbose=False,
               thin=1, burn=0, seed=None,
               return_logprobs=False):
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

    # an array to store the logprobs in (even if the user doesn't want them)
    logprob_track = np.empty(num_samples)

    # burn some samples if needed.
    if burn > 0:
        if verbose:
            print("burn-in sampling started")
        samples = sample_HMC(f, num_samples=burn, Lmin=Lmin, Lmax=Lmax,
                             epsilon=epsilon,
                             thin=1, burn=0)
        if verbose:
            print("burn-in sampling ended")
        x0 = samples[-1]

    def logprob_and_grad():
        logprob = tf.negative(model.objective)
        grads = tf.gradient(logprob, model.trainable_variables)
        return logprob, grads

    xs = _do_map(lambda x_i: x_i.read_value(), x)
    logprob, grads = logprob_and_grad()

    accept_count_batch = 0

    steps = num_samples * thin
    for i in range(1, steps):
        # make a copy of the old state.
        # grad_old = x.copy(), logprob, grad.copy()
        # p_old = RNG.randn(D)

        logprob, grads = logprob_and_grad()

        def init_ps_fn(x):
            return tf.random_normal(tf.shape(x), seed, dtype=x.dtype)
        ps_init = _do_map(init_ps, xs)

        def update_p_fn(args, p, grad, coeff=1):
            return p + coeff * epsilon * grad

        leapfrog_iter_max = tf.random_uniform(lmin, lmax, dtype=tf.int32, seed=seed)

        def leapfrog_update_x_fn(args, x, p):
            return x.assign_add(epsilon * p)

        def leapfrog_check_premature(grads):
            fins = _do_map(lambda tensor: tf.reduce_all(tf.is_finite(tensor)), grads)
            return tf.reduce_all(fins)

        def leapfrog_loop_cond_fn(i, early_stop, _):
            return not stop and i < leapfrog_iter_max

        def leapfrog_loop_body_fn(i, _stop, ps, _xs):
            new_xs = _do_map(leapfrog_update_x_fn, xs, ps)
            with tf.control_dependencies(new_xs):
                logprob, grads = logprob_and_grad()
                proceed = leapfrog_check_premature(grads)
                new_ps = tf.cond(proceed, _do_map(update_p, ps, grads), ps)
                return i + 1, proceed, new_ps, new_xs

        i = tf.convert_to_tensor(0, dtype=tf.int32)
        stop = tf.convert_to_tensor(False, dtype=tf.bool)
        ps = _do_map(update_p, ps_init, grads, coeff=0.5)
        _i, premature_reject, ps_loop, xs_loop = tf.while_loop(
            leapfrog_loop_cond,
            leapfrog_loop_body,
            [i, stop, ps, xs],
            parallel_iterations=1)

        tf.cond(premature_reject, normal_proposal())

        def normal_proposal():
            ps_new = _do_map(update_p, ps_loop, grad_loop, coeff=-0.5)

        # Standard HMC - begin leapfrogging
        premature_reject = False
        p = p_old + 0.5 * epsilon * grad
        for l in range(RNG.randint(Lmin, Lmax)):
            x += epsilon * p
            logprob, grad = f(x)
            logprob, grad = -logprob, -grad
            if np.any(np.isnan(grad)):  # pragma: no cover
                premature_reject = True
                break
            p += epsilon * grad
        p -= 0.5 * epsilon * grad
        # leapfrogging done

        # reject the proposal if there are numerical errors.
        if premature_reject:  # pragma: no cover
            print("warning: numerical instability.\
                  Rejecting this proposal prematurely")
            x, logprob, grad = x_old, logprob_old, grad_old
            if t % thin == 0:
                samples[t // thin] = x_old
                logprob_track[t // thin] = logprob_old
            continue

        # work out whether to accept the proposal
        def compute_proposal():
        log_accept_ratio = logprob - 0.5 * p.dot(p) - logprob_old + 0.5 * p_old.dot(p_old)
        logu = np.log(RNG.rand())

        if logu < log_accept_ratio:  # accept
            if t % thin == 0:
                samples[t // thin] = x
                logprob_track[t // thin] = logprob
            accept_count_batch += 1
        else:  # reject
            if t % thin == 0:
                samples[t // thin] = x_old
                logprob_track[t // thin] = logprob_old
            x, logprob, grad = x_old, logprob_old, grad_old
    if return_logprobs:
        return samples, logprob_track
    else:
        return samples

def _do_map(func, *args, **kwargs):
    return [func(*a, **kwargs) for a in zip(*args)]

def _leapfrog_steps(xs, xs_grad_prev, ps_prev, epsilon, lmin, lmax, seed=None):
    def build_ps(x, p_prev):
        return p_prev + 0.5 * epsilon * x
    ps = _do_map(build_ps, zip(xs_grad_prev, ps_prev))
    premature_reject = tf.convert_to_tensor(False, dtype=tf.bool)
    iter_max = tf.random_uniform(lmin, lmax, dtype=tf.int32, seed=seed)

    def cond(i, _ps):
        return tf.less(i, iter_max)

     = tf.while_loop(cond, body, loop_vars, back_prop=False)
    for l in range(RNG.randint(Lmin, Lmax)):
        x += epsilon * p
        logprob, grad = f(x)
        logprob, grad = -logprob, -grad
        if np.any(np.isnan(grad)):  # pragma: no cover
            premature_reject = True
            break
        p += epsilon * grad
    p -= 0.5 * epsilon * grad
    # leapfrogging done
