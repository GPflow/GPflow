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

    D = x0.size
    samples = np.zeros((num_samples, D))
    samples[0] = x0.copy()
    x = x0.copy()
    logprob = tf.negative(model.objective)
    grads = tf.gradient(logprob, model.trainable_variables)
    logprob_track[0] = logprob

    accept_count_batch = 0

    steps = num_samples * thin
    for i in range(1, steps):
        # make a copy of the old state.
        xs = _do_map(lambda x_i: x_i.read_value(), x)
        # grad_old = x.copy(), logprob, grad.copy()
        # p_old = RNG.randn(D)

        def random_ps(x):
            return tf.random_normal(tf.shape(x), seed, dtype=x.dtype)
        ps_prev = _do_map(random_ps, xs)

        ps = _do_map(lambda x: p_prev + 0.5 * epsilon * x, zip(xs_grad_prev, ps_prev))
        premature_reject = tf.convert_to_tensor(False, dtype=tf.bool)
        iter_max = tf.random_uniform(lmin, lmax, dtype=tf.int32, seed=seed)

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

def _do_map(tensors, func):
    return list(map(func, tensors))

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
