# Copyright 2019-2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Callable, Optional, Sequence, Tuple

import tensorflow as tf

from gpflow.base import Parameter

__all__ = ["SamplingHelper"]


class SamplingHelper:
    """
    This helper makes it easy to read from variables being set with a prior and
    writes values back to the same variables.

    Example::

        model = ...  # Create a GPflow model
        hmc_helper = SamplingHelper(model.log_posterior_density, model.trainable_parameters)

        target_log_prob_fn = hmc_helper.target_log_prob_fn
        current_state = hmc_helper.current_state

        hmc = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_log_prob_fn, ...)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(hmc, ...)

        @tf.function
        def run_chain_fn():
            return mcmc.sample_chain(
                num_samples, num_burnin_steps, current_state, kernel=adaptive_hmc)

        hmc_samples = run_chain_fn()
        parameter_samples = hmc_helper.convert_to_constrained_values(hmc_samples)
    """

    def __init__(
        self, target_log_prob_fn: Callable[[], tf.Tensor], parameters: Sequence[Parameter]
    ) -> None:
        """
        :param target_log_prob_fn: a callable which returns the log-density of the model
            under the target distribution; needs to implicitly depend on the `parameters`.
            E.g. `model.log_posterior_density`.
        :param parameters: List of :class:`gpflow.Parameter` used as a state of the Markov chain.
            E.g. `model.trainable_parameters`
            Note that each parameter must have been given a prior.
        """
        if not all(isinstance(p, Parameter) and p.prior is not None for p in parameters):
            raise ValueError(
                "`parameters` should only contain gpflow.Parameter objects with priors"
            )

        self._parameters = parameters
        self._target_log_prob_fn = target_log_prob_fn
        self._variables = [p.unconstrained_variable for p in parameters]

    @property
    def current_state(self) -> Sequence[tf.Variable]:
        """Return the current state of the unconstrained variables, used in HMC."""

        return self._variables

    @property
    def target_log_prob_fn(
        self,
    ) -> Callable[..., Tuple[tf.Tensor, Callable[..., Tuple[tf.Tensor, Sequence[None]]]]]:
        """
        The target log probability, adjusted to allow for optimisation to occur on the tracked
        unconstrained underlying variables.
        """
        variables_list = self.current_state

        @tf.custom_gradient
        def _target_log_prob_fn_closure(
            *variables: tf.Variable,
        ) -> Tuple[tf.Tensor, Callable[..., Tuple[tf.Tensor, Sequence[None]]]]:
            for v_old, v_new in zip(variables_list, variables):
                v_old.assign(v_new)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables_list)
                log_prob = self._target_log_prob_fn()
                # Now need to correct for the fact that the prob fn is evaluated on the
                # constrained space while we wish to evaluate it in the unconstrained space
                for param in self._parameters:
                    if param.transform is not None:
                        x = param.unconstrained_variable
                        log_det_jacobian = param.transform.forward_log_det_jacobian(
                            x, x.shape.ndims
                        )
                        log_prob += tf.reduce_sum(log_det_jacobian)

            @tf.function
            def grad_fn(
                dy: tf.Tensor, variables: Optional[tf.Tensor] = None
            ) -> Tuple[tf.Tensor, Sequence[None]]:
                grad = tape.gradient(log_prob, variables_list)
                return grad, [None] * len(variables_list)

            return log_prob, grad_fn

        return _target_log_prob_fn_closure  # type: ignore[no-any-return]

    def convert_to_constrained_values(
        self, hmc_samples: Sequence[tf.Tensor]
    ) -> Sequence[tf.Tensor]:
        """
        Converts list of unconstrained values in `hmc_samples` to constrained
        versions. Each value in the list corresponds to an entry in parameters
        passed to the constructor; for parameters that have a transform, the
        constrained representation is returned.
        """
        values = []
        for hmc_value, param in zip(hmc_samples, self._parameters):
            if param.transform is not None:
                value = param.transform.forward(hmc_value)
            else:
                value = hmc_value
            values.append(value)
        return values
