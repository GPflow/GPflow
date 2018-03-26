# Copyright 2017 Artem Artemev @awav
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

import itertools
import tensorflow as tf
import numpy as np
import pandas as pd

from .optimizer import Optimizer
from ..decors import name_scope

class HMC(Optimizer):
    def sample(self, model, num_samples, epsilon,
               lmin=1, lmax=1, thin=1, burn=0,
               session=None, initialize=True, anchor=True,
               logprobs=True):
        """
        A straight-forward HMC implementation. The mass matrix is assumed to be the
        identity.

        The gpflow model must implement `build_objective` method to build `f` function
        (tensor) which in turn based on model's internal trainable parameters `x`.

            f(x) = E(x)

        we then generate samples from the distribution

            pi(x) = exp(-E(x))/Z

        The total number of iterations is given by:

            burn + thin * num_samples

        The leafrog (Verlet) integrator works by picking a random number of steps
        uniformly between lmin and lmax, and taking steps of length epsilon.

        :param model: gpflow model with `build_objective` method implementation.
        :param num_samples: number of samples to generate.
        :param epsilon: HMC tuning parameter - stepsize.
        :param lmin: HMC tuning parameter - lowest integer `a` of uniform `[a, b]` distribution
            used for drawing number of leapfrog iterations.
        :param lmax: HMC tuning parameter - largest integer `b` from uniform `[a, b]` distribution
            used for drawing number of leapfrog iterations.
        :param thin: an integer which specifies the thinning interval.
        :param burn: an integer which specifies how many initial samples to discard.
        :param session: TensorFlow session. The default session or cached GPflow session
            will be used if it is none.
        :param initialize: indication either TensorFlow initialization is required or not.
        :param anchor: dump live trainable values computed within specified TensorFlow
            session to actual parameters (in python scope).
        :param logprobs: indicates either logprob values shall be included in output or not.

        :return: data frame with `num_samples` traces, where columns are full names of
            trainable parameters except last column, which is `logprobs`.
            Trainable parameters are represented as constrained values in output.

        :raises: ValueError exception in case when wrong parameter ranges were passed.
        """

        if lmax <= 0 or lmin <= 0:
            raise ValueError('The lmin and lmax parameters must be greater zero.')
        if thin <= 0:
            raise ValueError('The thin parameter must be greater zero.')
        if burn < 0:
            raise ValueError('The burn parameter must be equal or greater zero.')

        lmax += 1
        session = model.enquire_session(session)

        model.initialize(session=session, force=initialize)

        with tf.name_scope('hmc'):
            params = list(model.trainable_parameters)
            xs = list(model.trainable_tensors)

            def logprob_grads():
                logprob = tf.negative(model.build_objective())
                grads = tf.gradients(logprob, xs)
                return logprob, grads

            thin_args = [logprob_grads, xs, thin, epsilon, lmin, lmax]

            if burn > 0:
                burn_op = _burning(burn, *thin_args)
                session.run(burn_op, feed_dict=model.feeds)

            xs_dtypes = _map(lambda x: x.dtype, xs)
            logprob_dtype = model.objective.dtype
            dtypes = _flat(xs_dtypes, [logprob_dtype])
            indices = np.arange(num_samples)

            def map_body(_):
                xs_sample, logprob_sample = _thinning(*thin_args)
                return _flat(xs_sample, [logprob_sample])

            hmc_output = tf.map_fn(map_body, indices, dtype=dtypes,
                                   back_prop=False, parallel_iterations=1)
            with tf.control_dependencies(hmc_output):
                unconstrained_trace, logprob_trace = hmc_output[:-1], hmc_output[-1]
                constrained_trace = _map(lambda x, param: param.transform.forward_tensor(x),
                                         unconstrained_trace, params)
                hmc_output = constrained_trace + [logprob_trace]

        names = [param.pathname for param in params]
        raw_traces = session.run(hmc_output, feed_dict=model.feeds)

        if anchor:
            model.anchor(session)

        traces = dict(zip(names, map(list, raw_traces[:-1])))
        if logprobs:
            traces.update({'logprobs': raw_traces[-1]})
        return pd.DataFrame(traces)

    def make_optimize_tensor(self, model, session=None, var_list=None, **kwargs):
        raise NotImplementedError('HMC does not provide make_optimize_tensor method')

    def minimize(self, model, **kwargs):
        raise NotImplementedError('HMC does not provide minimize method, use `sample` instead.')


@name_scope("burning")
def _burning(burn, logprob_grads_fn, xs, *thin_args):
    def cond(i, _xs, _logprob):
        return i < burn

    def body(i, _xs, _logprob):
        xs_new, logprob = _thinning(logprob_grads_fn, xs, *thin_args)
        return i + 1, xs_new, logprob

    logprob, _grads = logprob_grads_fn()
    return _while_loop(cond, body, [0, xs, logprob])


@name_scope("thinning")
def _thinning(logprob_grads_fn, xs, thin, epsilon, lmin, lmax):
    def cond(i, _sample, _logprob, _grads):
        return i < thin

    def body(i, xs_copy, logprob_prev, grads_prev):
        ps_init = _init_ps(xs_copy)
        ps = _update_ps(ps_init, grads_prev, epsilon, coeff=+0.5)
        max_iters = tf.random_uniform((), minval=lmin, maxval=lmax, dtype=tf.int32)

        dep_list = _flat([max_iters], ps, ps_init)
        with tf.control_dependencies(dep_list):
            leapfrog_result = _leapfrog_step(xs, ps, epsilon, max_iters, logprob_grads_fn)
            proceed, xs_new, ps_new, logprob_new, grads_new = leapfrog_result
            dep_list = _flat([proceed], [logprob_new], xs_new, ps_new, grads_new)

            def standard_proposal():
                with tf.control_dependencies(dep_list):
                    return _reject_accept_proposal(
                        xs_new, xs_copy, ps_new, ps_init,
                        logprob_new, logprob_prev,
                        grads_new, grads_prev, epsilon)

            def premature_reject():
                with tf.control_dependencies(dep_list):
                    return _premature_reject(
                        xs_copy, logprob_prev, grads_prev)

            xs_out, logprob_out, grads_out = tf.cond(proceed,
                                                     standard_proposal,
                                                     premature_reject,
                                                     strict=True)

            xs_assign = _assign_variables(xs, xs_out)
            with tf.control_dependencies(xs_assign):
                xs_out_copy = _copy_variables(xs_assign)
                with tf.control_dependencies(xs_copy):
                    return i + 1, xs_out_copy, logprob_out, grads_out

    xs_in = _copy_variables(xs)
    logprob, grads = logprob_grads_fn()
    with tf.control_dependencies(_flat([logprob], xs_in, grads)):
        _, xs_res, logprob_res, _grads = _while_loop(cond, body, [0, xs_in, logprob, grads])
        return xs_res, logprob_res


@name_scope("premature_reject")
def _premature_reject(xs, logprob_prev, grads_prev):
    return xs, logprob_prev, grads_prev


@name_scope("reject_accept_proposal")
def _reject_accept_proposal(xs, xs_prev,
                            ps, ps_prev,
                            logprob, logprob_prev,
                            grads, grads_prev,
                            epsilon):
    def dot(ps_values):
        return tf.reduce_sum(_map(lambda p: tf.reduce_sum(tf.square(p)), ps_values))

    ps_upd = _update_ps(ps, grads, epsilon, coeff=-0.5)

    with tf.control_dependencies(ps_upd):
        log_accept_ratio = logprob - 0.5 * dot(ps_upd) - logprob_prev + 0.5 * dot(ps_prev)
        logu = tf.log(tf.random_uniform(shape=tf.shape(log_accept_ratio), dtype=logprob.dtype))

        def accept():
            with tf.control_dependencies([logu, log_accept_ratio]):
                return xs, logprob, grads

        def reject():
            with tf.control_dependencies([logu, log_accept_ratio]):
                return xs_prev, logprob_prev, grads_prev

        decision = logu < log_accept_ratio
        return tf.cond(decision, accept, reject, strict=True)


@name_scope("leapfrog")
def _leapfrog_step(xs, ps, epsilon, max_iterations, logprob_grads_fn):
    def update_xs(ps_values):
        return _map(lambda x, p: x.assign_add(epsilon * p), xs, ps_values)

    def whether_proceed(grads):
        finits = _map(lambda grad: tf.reduce_all(tf.is_finite(grad)), grads)
        return tf.reduce_all(finits)

    def cond(i, proceed, _ps, _xs):
        return tf.logical_and(proceed, i < max_iterations)

    def body(i, _proceed, ps, _xs):
        xs_new = update_xs(ps)
        with tf.control_dependencies(xs_new):
            _, grads = logprob_grads_fn()
            proceed = whether_proceed(grads)
            def ps_step():
                with tf.control_dependencies(grads):
                    return _update_ps(ps, grads, epsilon)
            def ps_no_step():
                with tf.control_dependencies(grads):
                    return ps

            ps_new = tf.cond(proceed, ps_step, ps_no_step, strict=True)
            return i + 1, proceed, ps_new, xs_new

    result = _while_loop(cond, body, [0, True, ps, xs])

    _i, proceed_out, ps_out, xs_out = result
    deps = _flat([proceed_out], ps_out, xs_out)
    with tf.control_dependencies(deps):
        logprob_out, grads_out = logprob_grads_fn()
        return proceed_out, xs_out, ps_out, logprob_out, grads_out


def _assign_variables(variables, values):
    return _map(lambda var, value: var.assign(value), variables, values)


def _copy_variables(variables):
    # NOTE: read_value with control_dependencies does not guarantee that
    #       that expected value will be returned.
    # return _map(lambda v: v.read_value(), variables)
    return _map(lambda var: var + 0, variables)


def _init_ps(xs):
    return _map(lambda x: tf.random_normal(tf.shape(x), dtype=x.dtype.as_numpy_dtype), xs)


def _update_ps(ps, grads, epsilon, coeff=1):
    return _map(lambda p, grad: p + coeff * epsilon * grad, ps, grads)


def _while_loop(cond, body, args):
    return tf.while_loop(cond, body, args, parallel_iterations=1, back_prop=False)


def _map(func, *args, **kwargs):
    return [func(*a, **kwargs) for a in zip(*args)]

def _flat(*elems):
    return list(itertools.chain.from_iterable(elems))
