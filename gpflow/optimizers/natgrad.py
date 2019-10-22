# Copyright 2018 Hugh Salimbeni, Artem Artemev @awav
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

import abc
import functools
from typing import Callable, List, Union, Tuple

import numpy as np
import tensorflow as tf

from ..base import Parameter

Scalar = Union[float, tf.Tensor, np.ndarray]


__all__ = [
    "NaturalGradient",
    "XiTransform",
    "XiSqrtMeanVar",
    "XiNat",
]


class XiTransform(metaclass=abc.ABCMeta):
    """
    XiTransform is the base class that implements three transformations necessary
    for the natural gradient calculation wrt any parameterization.
    This class does not handle any shape information, but it is assumed that
    the parameters pairs are always of shape (N, D) and (D, N, N).
    """

    @abc.abstractmethod
    def meanvarsqrt_to_xi(self, mean, varsqrt):
        """
        Transforms the parameter `mean` and `varsqrt` to `xi_1`, `xi_2`

        :param mean: the mean parameter (N, D)
        :param varsqrt: the varsqrt parameter (N, N, D)
        :return: tuple (xi_1, xi_2), the xi parameters (N, D), (N, N, D)
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        """
        Transforms the parameter `xi_1`, `xi_2` to `mean`, `varsqrt`

        :param xi_1: the xi_1 parameter
        :param xi_2: the xi_2 parameter
        :return: tuple (mean, varsqrt), the meanvarsqrt parameters
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def naturals_to_xi(self, nat_1, nat_2):
        """
        Applies the transform so that `nat_1`, `nat_2` is mapped to `xi_1`, `xi_2`

        :param nat_1: the nat_1 parameter
        :param nat_2: the nat_1 parameter
        :return: tuple `xi_1`, `xi_2`
        """
        pass  # pragma: no cover


class NaturalGradient(tf.optimizers.Optimizer):
    def __init__(self, gamma: Scalar, name=None):
        name = self.__class__.__name__ if name is None else name
        super().__init__(name)
        self.gamma = gamma

    def minimize(self, loss_fn: Callable, var_list: List[Parameter]):
        """
        Minimizes objective function of the model.
        Natural Gradient optimizer works with variational parameters only.
        There are two supported ways of transformation for parameters:
            - XiNat
            - XiSqrtMeanVar
        Custom transformations are also possible, they should implement
        `XiTransform` interface.

            :param model: GPflow model.
            :param session: Tensorflow session where optimization will be run.
            :param var_list: List of pair tuples of variational parameters or
                triplet tuple with variational parameters and ξ transformation.
                By default, all parameters goes through XiNat() transformation.
                For example your `var_list` can look as,
                ```
                var_list = [
                    (q_mu1, q_sqrt1),
                    (q_mu2, q_sqrt2, XiSqrtMeanVar())
                ]
                ```
            :param feed_dict: Feed dictionary of tensors passed to session run method.
            :param maxiter: Number of run interation. Default value: 1000.
            :param anchor: Synchronize updated parameters for a session with internal
                parameter's values.
            :param step_callback: A callback function to execute at each optimization step.
                The callback should accept variable argument list, where first argument is
                optimization step number.
            :type step_callback: Callable[[], None]
            :param kwargs: Extra parameters passed to session run's method.
        """
        parameters = [(v[0], v[1], v[2] if len(v) > 2 else XiNat()) for v in var_list]
        self._natgrad_steps(loss_fn, parameters)

    def _natgrad_steps(self, loss_fn: Callable, parameters: List[Tuple[Parameter, Parameter, XiTransform]]):
        def natural_gradient_step(q_mu, q_sqrt, xi_transform):
            self._natgrad_step(loss_fn, q_mu, q_sqrt, xi_transform)

        with tf.name_scope("natural_gradient_steps"):
            list(map(natural_gradient_step, *zip(*parameters)))

    def _natgrad_step(self, loss_fn: Callable, q_mu: Parameter, q_sqrt: Parameter, xi_transform: XiTransform):
        """
        Implements equation [10] from

        @inproceedings{salimbeni18,
            title={Natural Gradients in Practice: Non-Conjugate  Variational Inference in Gaussian Process Models},
            author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
            booktitle={AISTATS},
            year={2018}

        In addition, for convenience with the rest of GPflow, this code computes ∂L/∂η using
        the chain rule:

        ∂L/∂η = (∂[q_μ, q_sqrt] / ∂η)(∂L / ∂[q_μ, q_sqrt])

        In total there are three derivative calculations:
        natgrad L w.r.t ξ  = (∂ξ / ∂nat) [ (∂[q_μ, q_sqrt] / ∂η)(∂L / ∂[q_μ, q_sqrt]) ]^T

        Note that if ξ = nat or [q_μ, q_sqrt] some of these calculations are the identity.

        """

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch([
                q_mu.unconstrained_variable,
                q_sqrt.unconstrained_variable
            ])
            loss = loss_fn()

            # the three parameterizations as functions of [q_mu, q_sqrt]
            eta1, eta2 = meanvarsqrt_to_expectation(q_mu, q_sqrt)

            # we need these to calculate the relevant gradients
            meanvarsqrt = expectation_to_meanvarsqrt(eta1, eta2)

            if not isinstance(xi_transform, XiNat):
                nat1, nat2 = meanvarsqrt_to_natural(q_mu, q_sqrt)
                xi1_nat, xi2_nat = xi_transform.naturals_to_xi(nat1, nat2)
                dummy_tensors = tf.ones_like(xi1_nat), tf.ones_like(xi2_nat)
                with tf.GradientTape(watch_accessed_variables=False) as forward_tape:
                    forward_tape.watch(dummy_tensors)
                    dummy_gradients = tape.gradient([xi1_nat, xi2_nat], [nat1, nat2], output_gradients=dummy_tensors)

        # 1) the oridinary gpflow gradient
        dL_dmean, dL_dvarsqrt = tape.gradient(loss, [q_mu, q_sqrt])
        dL_dvarsqrt = q_sqrt.transform.forward(dL_dvarsqrt)

        # 2) the chain rule to get ∂L/∂η, where η (eta) are the expectation parameters
        dL_deta1, dL_deta2 = tape.gradient(meanvarsqrt, [eta1, eta2], output_gradients=[dL_dmean, dL_dvarsqrt])

        if not isinstance(xi_transform, XiNat):
            nat_dL_xi1, nat_dL_xi2 = forward_tape.gradient(dummy_gradients,
                                                           dummy_tensors,
                                                           output_gradients=[dL_deta1, dL_deta2])
        else:
            nat_dL_xi1, nat_dL_xi2 = dL_deta1, dL_deta2

        xi1, xi2 = xi_transform.meanvarsqrt_to_xi(q_mu, q_sqrt)
        xi1_new = xi1 - self.gamma * nat_dL_xi1
        xi2_new = xi2 - self.gamma * nat_dL_xi2

        # Transform back to the model parameters [q_μ, q_sqrt]
        mean_new, varsqrt_new = xi_transform.xi_to_meanvarsqrt(xi1_new, xi2_new)

        q_mu.assign(mean_new)
        q_sqrt.assign(varsqrt_new)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self._serialize_hyperparameter('gamma'),
        })
        return config


#
# Xi transformations necessary for natural gradient optimizer.
# Abstract class and two implementations: XiNat and XiSqrtMeanVar.
#


class XiNat(XiTransform):
    """
    This is the default transform. Using the natural directly saves the forward mode
     gradient, and also gives the analytic optimal solution for gamma=1 in the case
     of Gaussian likelihood.
    """

    def meanvarsqrt_to_xi(self, mean, varsqrt):
        return meanvarsqrt_to_natural(mean, varsqrt)

    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        return natural_to_meanvarsqrt(xi_1, xi_2)

    def naturals_to_xi(self, nat_1, nat_2):
        return nat_1, nat_2


class XiSqrtMeanVar(XiTransform):
    """
    This transformation will perform natural gradient descent on the model parameters,
    so saves the conversion to and from Xi.
    """

    def meanvarsqrt_to_xi(self, mean, varsqrt):
        return mean, varsqrt

    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        return xi_1, xi_2

    def naturals_to_xi(self, nat_1, nat_2):
        return natural_to_meanvarsqrt(nat_1, nat_2)


#
# Auxiliary gaussian parameter conversion functions.
#
# The following functions expect their first and second inputs to have shape
# DN1 and DNN, respectively. Return values are also of shapes DN1 and DNN.


def swap_dimensions(method):
    """
    Converts between GPflow indexing and tensorflow indexing
    `method` is a function that broadcasts over the first dimension (i.e. like all tensorflow matrix ops):
        `method` inputs DN1, DNN
        `method` outputs DN1, DNN
    :return: Function that broadcasts over the final dimension (i.e. compatible with GPflow):
        inputs: ND, DNN
        outputs: ND, DNN
    """

    @functools.wraps(method)
    def wrapper(a_nd, b_dnn, swap=True):
        if swap:
            if a_nd.shape.ndims != 2:  # pragma: no cover
                raise ValueError("The `a_nd` input must have 2 dimensions.")
            a_dn1 = tf.linalg.adjoint(a_nd)[:, :, None]
            A_dn1, B_dnn = method(a_dn1, b_dnn)
            A_nd = tf.linalg.adjoint(A_dn1[:, :, 0])
            return A_nd, B_dnn
        else:
            return method(a_nd, b_dnn)

    return wrapper


@swap_dimensions
def natural_to_meanvarsqrt(nat_1, nat_2):
    var_sqrt_inv = tf.linalg.cholesky(-2 * nat_2)
    var_sqrt = _inverse_lower_triangular(var_sqrt_inv)
    S = tf.linalg.matmul(var_sqrt, var_sqrt, transpose_a=True)
    mu = tf.linalg.matmul(S, nat_1)
    # We need the decomposition of S as L L^T, not as L^T L,
    # hence we need another cholesky.
    return mu, tf.linalg.cholesky(S)


@swap_dimensions
def meanvarsqrt_to_natural(mu: tf.Tensor, s_sqrt: tf.Tensor):
    s_sqrt_inv = _inverse_lower_triangular(s_sqrt)
    s_inv = tf.linalg.matmul(s_sqrt_inv, s_sqrt_inv, transpose_a=True)
    return tf.linalg.matmul(s_inv, mu), -0.5 * s_inv


@swap_dimensions
def natural_to_expectation(nat_1: tf.Tensor, nat_2: tf.Tensor):
    return meanvarsqrt_to_expectation(*natural_to_meanvarsqrt(nat_1, nat_2, swap=False), swap=False)


@swap_dimensions
def expectation_to_natural(eta_1: tf.Tensor, eta_2: tf.Tensor):
    return meanvarsqrt_to_natural(*expectation_to_meanvarsqrt(eta_1, eta_2, swap=False), swap=False)


@swap_dimensions
def expectation_to_meanvarsqrt(eta_1, eta_2):
    var = eta_2 - tf.linalg.matmul(eta_1, eta_1, transpose_b=True)
    return eta_1, tf.linalg.cholesky(var)


@swap_dimensions
def meanvarsqrt_to_expectation(m: tf.Tensor, v_sqrt: tf.Tensor):
    v = tf.linalg.matmul(v_sqrt, v_sqrt, transpose_b=True)
    return m, v + tf.linalg.matmul(m, m, transpose_b=True)


def _inverse_lower_triangular(M):
    """
    Take inverse of lower triangular (e.g. Cholesky) matrix. This function
    broadcasts over the first index.

    :param M: Tensor with lower triangular structure of shape DxNxN
    :return: The inverse of the Cholesky decomposition. Same shape as input.
    """
    if M.shape.ndims != 3:  # pragma: no cover
        raise ValueError("Number of dimensions for input is required to be 3.")
    D, N = M.shape[0], M.shape[1]
    I_DNN = tf.eye(N, dtype=M.dtype)[None, :, :] * tf.ones((D, 1, 1), dtype=M.dtype)
    return tf.linalg.triangular_solve(M, I_DNN)
