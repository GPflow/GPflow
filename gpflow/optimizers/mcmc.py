# Copyright 2019 Artem Artemev @awav
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

from dataclasses import dataclass
from typing import Callable, List, Optional, TypeVar

import tensorflow as tf

import gpflow

__all__ = ["positive_parameter", "SamplingHelper"]


def positive_parameter(value: tf.Tensor):
    if isinstance(value, (tf.Variable, gpflow.Parameter)):
        return value
    return gpflow.Parameter(value, transform=gpflow.positive())


ModelParameters = List[TypeVar("ModelParameter", tf.Variable, gpflow.Parameter)]


@dataclass(frozen=True)
class SamplingHelper:
    """
    Helper reads from variables being set with a prior and writes values back to the same variables.

    Args:
        target_log_prob_fn: Python callable which represents log-density under the target distribution.
        parameters: List of `Variable`'s or gpflow `Parameter`s used as a state of the Markov chain.
    """

    target_log_prob_fn: Callable[[ModelParameters], tf.Tensor]
    parameters: ModelParameters

    @property
    def variables(self):
        """
        Returns the same list of parameters as `parameters` property, but replaces gpflow `Parameter`s
        with their unconstrained variables - `parameter.unconstrained_variable`.
        """
        return [p.unconstrained_variable if isinstance(p, gpflow.Parameter) else p for p in self.parameters]

    def assign_values(self, *values, unconstrained: Optional[bool] = True):
        """
        Assings (constrained or unconstrained) values to the parameter's variable.
        Unconstrained values are assigned to the list of `variables` property.
        """
        trainables = self.variables if unconstrained else self.parameters
        assert len(values) == len(trainables)
        n = len(trainables)
        for i in range(n):
            trainables[i].assign(values[i])

    def convert_to_constrained_values(self, *unconstrained_values):
        """
        Converts list of `unconstrained_values` to constrained versions. Each value in the list correspond to the
        parameter and in case when an object in the same position has `gpflow.Parameter` type, the `forward` method
        of transform will be applied.
        """
        samples = []
        for i, values in enumerate(unconstrained_values):
            param = self.parameters[i]
            if isinstance(param, gpflow.Parameter) and param.transform is not None:
                sample = param.transform.forward(values)
            else:
                sample = values
            samples.append(sample.numpy())
        return samples

    def make_posterior_log_prob_fn(self):
        """
        Make a differentiable posterior log-probability function using helper's `target_log_prob_fn` with respect to
        passed `parameters`.
        """

        @tf.custom_gradient
        def log_prob_fn(*values):
            self.assign_values(*values)

            variables_to_watch = self.variables
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(variables_to_watch)
                log_prob = self.target_log_prob_fn()

            @tf.function
            def grad_fn(in_grad: tf.Tensor, variables: Optional[tf.Variable] = None):
                grad = tape.gradient(log_prob, variables_to_watch)
                return grad, [None] * len(variables)

            return log_prob, grad_fn

        return log_prob_fn
