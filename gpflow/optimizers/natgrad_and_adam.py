# Copyright 2018-2021 The GPflow Contributors, Amazon.com. All Rights Reserved.
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

from typing import Callable, Sequence, Tuple, Union

import tensorflow as tf

from gpflow.base import Parameter

from .natgrad import LossClosure, NatGradParameters, Scalar, XiNat, natgrad_apply_gradients

TensorflowLearningRate = Union[
    Scalar, tf.Tensor, tf.keras.optimizers.schedules.LearningRateSchedule, Callable[[], Scalar]
]

__all__ = [
    "JointNaturalGradientAndAdam",
]


class JointNaturalGradientAndAdam(tf.optimizers.Optimizer):
    """
    Implements a combined Natural Gradient and Adam optimizer. This calculates
    gradients for variational parameters as well as non-variational parameters in a single
    backwards pass (as opposed to using two backward passes for NatGrad and Adam separately).
    Then, it applies the NatGrad update on the variational parameters, and the Adam update on the
    non-variational parameters.

    Note that the behaviour of this optimizer is slightly different to the usual NatGrad+Adam behaviour
    when done in two different steps. With this optimizer, taking a NatGrad step of gamma=1.0 for the Gaussian
    likelihood case will not lead to the optimal inducing locations as the kernel hyperparameters would've
    changed. However, any other potential effects on convergence will need to be evaluated empirically.

    Note that this optimizer does not implement the standard API of
    tf.optimizers.Optimizer. Its only public method is minimize(), which has
    a custom signature (var_list needs to be a list of (q_mu, q_sqrt) tuples,
    where q_mu and q_sqrt are gpflow.Parameter instances, not tf.Variable).

    Note furthermore that the natural gradients are implemented only for the
    full covariance case (i.e., q_diag=True is NOT supported).

    When using in your work, please cite
        @inproceedings{salimbeni18,
            title={Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models},
            author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
            booktitle={AISTATS},
            year={2018}
    """

    def __init__(self, gamma: Scalar, adam_lr: TensorflowLearningRate, name=None):
        """
        :param gamma: natgrad step length
        :param adam_lr: adam learning rate
        """
        name = self.__class__.__name__ if name is None else name
        super().__init__(name)
        self.gamma = gamma
        self.xi_transform = XiNat()

        self.adam_lr = adam_lr
        self.adam_optimizer = tf.optimizers.Adam(learning_rate=self.adam_lr)

    def minimize(
        self,
        loss_fn: LossClosure,
        variational_var_list: Sequence[NatGradParameters],
        non_variational_var_list: Sequence[Parameter],
    ):
        """
        Minimizes objective function of the model.
        Natural Gradient optimizer works with variational parameters only.

        :param loss_fn: Loss function.
        :param variational_var_list: List of pair tuples of variational parameters or
            triplet tuple with variational parameters. These parameters will be optimized
            using NatGrad.

            For example, `var_list` could be
            ```
            var_list = [
                (q_mu1, q_sqrt1),
                (q_mu2, q_sqrt2)
            ]
            ```
        :param non_variational_var_list: List of non-variational parameters to be optimized
            using Adam.
        """
        variational_parameters = [(v[0], v[1]) for v in variational_var_list]
        self._natgrad_steps(loss_fn, variational_parameters, non_variational_var_list)

    def _natgrad_steps(
        self,
        loss_fn: LossClosure,
        variational_parameters: Sequence[Tuple[Parameter, Parameter]],
        non_variational_var_list: Sequence[Parameter],
    ):
        """
        Computes gradients of loss_fn() w.r.t. variational parameters and
        nonvariational parameters in a single backward pass. Then, it updates
        the variational parameters using natgrad, and the non-variational parameters
        using Adam.

        :param loss_fn: Loss function.
        :param variational_parameters: List of tuples (q_mu, q_sqrt)
        :param non_variational_var_list: List of non-variational parameters to be optimized
            using Adam.
        """
        q_mus, q_sqrts = zip(*variational_parameters)
        q_mu_vars = [p.unconstrained_variable for p in q_mus]
        q_sqrt_vars = [p.unconstrained_variable for p in q_sqrts]

        # Calculate loss whilst doing backprop on both variational and non-variational parameters.
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(q_mu_vars + q_sqrt_vars + list(non_variational_var_list))
            training_loss = loss_fn()

        # Take Adam Step on non-variational parameters
        gradients = tape.gradient(training_loss, non_variational_var_list)
        self.adam_optimizer.apply_gradients(zip(gradients, non_variational_var_list))

        # Take NatGrad step on variational parameters
        q_mu_grads, q_sqrt_grads = tape.gradient(training_loss, [q_mu_vars, q_sqrt_vars])

        del tape  # Remove "persistent" tape

        with tf.name_scope(f"{self._name}/natural_gradient_steps"):
            for q_mu_grad, q_sqrt_grad, q_mu, q_sqrt in zip(
                q_mu_grads, q_sqrt_grads, q_mus, q_sqrts
            ):
                natgrad_apply_gradients(
                    self.gamma, q_mu_grad, q_sqrt_grad, q_mu, q_sqrt, self.xi_transform
                )

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self._serialize_hyperparameter("gamma")})
        config.update({"adam_lr": self._serialize_hyperparameter("adam_lr")})
        return config
