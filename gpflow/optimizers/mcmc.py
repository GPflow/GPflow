# Copyright 2019 Artem Artemev @awav, Eric Hambro @condnsdmatters
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

from typing import Callable, Sequence, Optional, TypeVar

import tensorflow as tf

from gpflow.base import Parameter

__all__ = ["SamplingHelper"]


ModelParameters = Sequence[TypeVar("ModelParameter", tf.Variable, Parameter)]
LogProbabilityFunction = Callable[[ModelParameters], tf.Tensor]


class SamplingHelper:
    """
    Helper reads from variables being set with a prior and writes values back to the same variables.

    Example:
        model = <Create GPflow model>
        hmc_helper = SamplingHelper(model.log_marginal_likelihood, m.trainable_parameters)

        target_log_prob_fn = hmc_helper.target_log_prob_fn
        current_state = hmc_helper.current_state

        hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_log_prob_fn, ...)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(hmc, ...)

        @tf.function
        def run_chain_fn():
            return mcmc.sample_chain(num_samples, num_burnin_steps, current_state, kernel=adaptive_hmc)

        hmc_samples = run_chain_fn()
        parameter_samples = hmc_helper.convert_samples_to_parameter_values(hmc_samples)

    Args:
        target_log_prob_fn: Python callable which represents log-density under the target distribution.
        model_parameters: List of `tensorflow.Variable`s or `gpflow.Parameter`s used as a state of the Markov chain.
    """

    def __init__(
        self, target_log_prob_fn: LogProbabilityFunction, model_parameters: ModelParameters
    ):
        assert all([isinstance(p, (Parameter, tf.Variable)) for p in model_parameters])

        self._model_parameters = model_parameters
        self._target_log_prob_fn = target_log_prob_fn

        self._parameters = []
        self._unconstrained_variables = []
        for p in self._model_parameters:
            self._unconstrained_variables.append(
                p.unconstrained_variable if isinstance(p, Parameter) else p
            )
            self._parameters.append(p)

    @property
    def current_state(self):
        """Return the current state of the unconstrained variables, used in HMC."""

        return self._unconstrained_variables

    @property
    def target_log_prob_fn(self):
        """ The target log probability, adjusted to allow for optimisation to occur on the tracked
        unconstrained underlying variables.
        """
        variables_list = self.current_state

        @tf.custom_gradient
        def _target_log_prob_fn_closure(*variables):
            for v_old, v_new in zip(variables_list, variables):
                v_old.assign(v_new)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables_list)
                log_prob = self._target_log_prob_fn()
                # Now need to correct for the fact that the prob fn is evaluated on the
                # constrained space while we wish to evaluate it in the unconstrained space
                for param in self._model_parameters:
                    if isinstance(param, Parameter) and param.transform is not None:
                        x = param.unconstrained_variable
                        log_det_jacobian = param.transform.forward_log_det_jacobian(
                            x, x.shape.ndims
                        )
                        log_prob += tf.reduce_sum(log_det_jacobian)

            @tf.function
            def grad_fn(dy, variables: Optional[tf.Tensor] = None):
                grad = tape.gradient(log_prob, variables_list)
                return grad, [None] * len(variables)

            return log_prob, grad_fn

        return _target_log_prob_fn_closure

    def convert_constrained_values(self, hmc_samples):
        """
        Converts list of `unconstrained_values` to constrained versions. Each value in the list correspond to an entry in
        `self.parameters`; in case that object is a `gpflow.Parameter`, the `forward` method of its transform
        will be applied first.
        """
        values = []
        for hmc_variable, param in zip(hmc_samples, self._parameters):
            if isinstance(param, Parameter) and param.transform is not None:
                value = param.transform.forward(hmc_variable)
            else:
                value = hmc_variable
            values.append(value.numpy())
        return values
